"""요청 단위 상태: 요청 ID, LLM 시간 예산(deadline), 토큰 사용량(usage).

ThreadPoolExecutor로 LLM을 동시에 부를 때는 반드시 submit()을 써야
요청 ID·예산·사용량이 작업 스레드까지 이어진다.
"""
from __future__ import annotations

import contextvars
import threading
import time
from concurrent.futures import Executor, Future
from typing import Any, Callable

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
_deadline_var: contextvars.ContextVar[float | None] = contextvars.ContextVar("deadline", default=None)
_usage_var: contextvars.ContextVar["UsageCollector | None"] = contextvars.ContextVar("usage", default=None)
# 이 작업이 속한 병렬 묶음(gather)들의 중단 신호. 바깥 묶음이 중단되면 안쪽 묶음도 멈춘다.
_cancel_var: contextvars.ContextVar[tuple[threading.Event, ...]] = contextvars.ContextVar("cancel", default=())


class UsageCollector:
    def __init__(self) -> None:
        self._items: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def add(self, role: str, model: str, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self._items.append({"role": role, "model": model,
                                "input_tokens": int(input_tokens), "output_tokens": int(output_tokens)})

    def items(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._items)


def begin(budget_seconds: float) -> UsageCollector:
    """엔드포인트 함수 맨 앞에서 부른다: 이 요청의 LLM 시간 예산과 사용량 기록을 시작."""
    _deadline_var.set(time.monotonic() + budget_seconds)
    collector = UsageCollector()
    _usage_var.set(collector)
    return collector


def remaining_seconds() -> float | None:
    deadline = _deadline_var.get()
    return None if deadline is None else deadline - time.monotonic()


def usage() -> UsageCollector:
    collector = _usage_var.get()
    if collector is None:
        collector = UsageCollector()
        _usage_var.set(collector)
    return collector


def submit(pool: Executor, fn: Callable[..., Any], *args: Any) -> Future:
    ctx = contextvars.copy_context()
    return pool.submit(ctx.run, fn, *args)


def cancelled() -> bool:
    """같은 병렬 묶음의 다른 작업이 실패해서, 이 작업의 결과가 더는 쓰이지 않는지."""
    return any(event.is_set() for event in _cancel_var.get())


def gather(pool: Executor, calls: list[tuple[Callable[..., Any], tuple]]) -> list[Any]:
    """calls를 pool에서 동시에 돌리고 결과를 순서대로 돌려준다.

    하나라도 실패하면 아직 시작 안 한 작업은 취소하고, 이미 돌고 있는 작업은 다음 LLM 호출(재시도 포함)을
    하지 않게 중단 신호를 건 다음, 첫 예외를 그대로 올린다. 이미 나가 있는 LLM 호출 하나는 끊을 수 없어서
    with ThreadPoolExecutor가 그 호출이 끝날 때까지는 기다린다 (호출마다 시간 예산 안에서 끝남).
    """
    stop = threading.Event()
    parent = _cancel_var.get()

    def run(fn: Callable[..., Any], args: tuple) -> Any:
        _cancel_var.set(parent + (stop,))   # submit이 복사한 컨텍스트 안에서만 바뀐다
        return fn(*args)

    futures = [submit(pool, run, fn, args) for fn, args in calls]
    try:
        return [f.result() for f in futures]
    except BaseException:
        stop.set()
        for f in futures:
            f.cancel()
        raise
