"""작업(Task) → 등급(light/deep/judge) → 공급자·모델, 그리고 '검증 후 한 번 다시 부르기'.

흐름: 호출 → JSON 파싱 → Pydantic(Enum 포함) 검증 → 근거 검증(validate)
     → 문제가 있으면 무엇이 틀렸는지 알려주고 다시 부름 (기본 1번, API 명세)
     → 그래도 실패하면 근거가 확인된 부분만 쓰거나(accept_after_retries) 502 LLM_ERROR

시간 예산: 요청마다 context.begin()으로 정한 예산(실시간 25초, 백그라운드 170초) 안에서만 부른다.
로그: llm_call(호출마다), llm_invalid_output(형식·근거 검증 실패, raw_output 앞 2,000자),
      llm_grounding_detail(근거 검증 실패 이유, 15건·건당 300자까지. LOG_RAW_OUTPUT=false면 안 찍음)
"""
from __future__ import annotations

import contextvars
import importlib
import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from ..config import Settings, get_settings
from ..context import cancelled, remaining_seconds, usage
from ..errors import LLMError, LLMTimeout, OutputTruncated
from ..logging_setup import log_event
from ..prompts import render
from .base import ImageInput, LLMProvider, LLMResult
from .json_schema import strict_schema

T = TypeVar("T", bound=BaseModel)
V = TypeVar("V")

MIN_CALL_SECONDS = 3.0      # 남은 시간이 이보다 적으면 새 호출을 시작하지 않음
DEFAULT_CALL_TIMEOUT = 170.0


class Task(str, Enum):
    IMAGE_READ = "image_read"
    PAGE_READ = "page_read"
    TERM_EXTRACT = "term_extract"
    TERM_VERIFY = "term_verify"
    SUMMARY_VERIFY = "summary_verify"
    NOTE_VERIFY = "note_verify"
    SOLVE_VERIFY = "solve_verify"
    TERM_SUMMARY = "term_summary"
    RELATIONS = "relations"
    TERM_JUDGE = "term_judge"
    HAND_RAISE = "hand_raise"
    QUESTION_MAKE = "question_make"
    NOTE = "note"
    EXAM_GENERATE = "exam_generate"
    EXAM_VERIFY = "exam_verify"
    EXAM_SOLVE = "exam_solve"
    EXAM_GRADE = "exam_grade"
    REPORT = "report"
    REPORT_VERIFY = "report_verify"


TASK_TIER: dict[Task, str] = {
    Task.IMAGE_READ: "light",
    Task.PAGE_READ: "light",
    Task.TERM_EXTRACT: "deep",
    Task.TERM_VERIFY: "deep",
    Task.SUMMARY_VERIFY: "deep",
    Task.NOTE_VERIFY: "deep",
    Task.SOLVE_VERIFY: "deep",
    Task.TERM_SUMMARY: "deep",
    Task.RELATIONS: "deep",
    Task.TERM_JUDGE: "judge",
    Task.EXAM_GENERATE: "deep",
    Task.EXAM_VERIFY: "deep",
    Task.EXAM_GRADE: "deep",
    Task.HAND_RAISE: "light",
    Task.QUESTION_MAKE: "light",
    Task.NOTE: "light",
    Task.EXAM_SOLVE: "light",
    Task.REPORT: "light",
    Task.REPORT_VERIFY: "deep",
}

# API 명세의 usage.role
TASK_ROLE: dict[Task, str] = {
    Task.IMAGE_READ: "material",
    Task.PAGE_READ: "material",
    Task.TERM_EXTRACT: "material",
    Task.TERM_VERIFY: "material",
    Task.SUMMARY_VERIFY: "material",
    Task.NOTE_VERIFY: "student",
    Task.SOLVE_VERIFY: "student",
    Task.TERM_SUMMARY: "material",
    Task.RELATIONS: "material",
    Task.TERM_JUDGE: "judge",
    Task.HAND_RAISE: "student",
    Task.QUESTION_MAKE: "student",
    Task.NOTE: "student",
    Task.EXAM_SOLVE: "student",
    Task.EXAM_GENERATE: "evaluator",
    Task.EXAM_VERIFY: "evaluator",
    Task.EXAM_GRADE: "evaluator",
    Task.REPORT: "report",
    Task.REPORT_VERIFY: "report",
}


@dataclass
class Outcome(Generic[V]):
    """validate의 결과. fatal이 있으면 다시 부르고, warnings는 그대로 통과."""
    value: V
    fatal: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class OutputRejected(Exception):
    """다시 불러도 출력이 검증을 통과하지 못함 → 502 LLM_ERROR로 바뀐다."""


# validate가 '이번이 마지막 시도인가'를 알 수 있게 한다. validate가 불린 횟수를 직접 세면
# 앞선 시도가 JSON 형식 오류·잘림으로 끝났을 때 (validate가 안 불려서) 계산이 어긋난다.
_last_attempt: contextvars.ContextVar[bool] = contextvars.ContextVar("llm_last_attempt", default=True)


def is_last_attempt() -> bool:
    """validate 안에서 부르면, 지금 검사하는 응답이 call_structured의 마지막 시도인지 돌려준다."""
    return _last_attempt.get()


_providers: dict[str, LLMProvider] = {}


def get_provider(name: str, settings: Settings | None = None) -> LLMProvider:
    if name in _providers:
        return _providers[name]
    s = settings or get_settings()
    try:   # 배포판에는 그 판에서 쓰는 공급자 파일만 들어 있다 (app/llm/{name}_provider.py)
        module = importlib.import_module(f".{name}_provider", __package__)
    except ModuleNotFoundError as exc:
        raise LLMError(f"공급자 파일이 없습니다: {name}_provider.py") from exc
    provider: LLMProvider = module.create(s)
    _providers[name] = provider
    return provider


def _time_left() -> float:
    left = remaining_seconds()
    if left is None:
        return DEFAULT_CALL_TIMEOUT
    if left < MIN_CALL_SECONDS:
        raise LLMTimeout("요청 시간 예산을 모두 썼습니다.")
    return left


# ── 호출 간격 (무료 등급용): 공급자마다 호출 사이를 LLM_MIN_INTERVAL_SECONDS만큼 띄운다 ──
_throttle_lock = threading.Lock()
_next_call_at: dict[str, float] = {}

TRANSIENT_BACKOFF = [4.0, 8.0, 15.0, 25.0]   # 공급자가 대기 시간을 안 알려줄 때
MAX_TRANSIENT_RETRIES = len(TRANSIENT_BACKOFF)


def _wait_turn(provider_name: str, interval: float) -> None:
    if interval <= 0:
        return
    with _throttle_lock:
        now = time.monotonic()
        start = max(now, _next_call_at.get(provider_name, 0.0))
        _next_call_at[provider_name] = start + interval
    delay = start - now
    if delay > 0:
        left = remaining_seconds()
        if left is not None and delay > left - MIN_CALL_SECONDS:
            raise LLMTimeout("호출 간격을 기다리면 시간 예산을 넘습니다.")
        time.sleep(delay)


# ── 서버 전체 동시 호출 상한: 요청이 여러 개 동시에 와도 LLM을 동시에 부르는 수를 LLM_GLOBAL_CONCURRENCY로 묶는다 ──
_global_slots: threading.BoundedSemaphore | None = None
_global_slots_size = -1
_global_lock = threading.Lock()


def _slots(size: int) -> threading.BoundedSemaphore | None:
    global _global_slots, _global_slots_size
    if size <= 0:
        return None
    with _global_lock:
        if _global_slots is None or _global_slots_size != size:
            _global_slots, _global_slots_size = threading.BoundedSemaphore(size), size
        return _global_slots


def _raw(text: str) -> dict:
    """llm_invalid_output의 raw_output (명세 필드 이름 유지). LOG_RAW_OUTPUT=false면 원문 대신 길이만 적는다."""
    if get_settings().log_raw_output:
        return {"raw_output": text[:2000]}
    return {"raw_output": f"[생략: {len(text)}자]"}


def _call_with_slot(provider: LLMProvider, settings: Settings, **kwargs) -> LLMResult:
    slots = _slots(settings.llm_global_concurrency)
    if slots is None:
        return provider.generate_json(timeout=_time_left(), **kwargs)
    wait = _time_left() - MIN_CALL_SECONDS
    if not slots.acquire(timeout=max(0.0, wait)):
        raise LLMTimeout("다른 요청의 LLM 호출이 많아 시간 예산 안에 차례가 오지 않았습니다.")
    try:
        return provider.generate_json(timeout=_time_left(), **kwargs)
    finally:
        slots.release()


def _generate(provider: LLMProvider, *, role: str, model: str, **kwargs) -> LLMResult:
    """한 번의 생성. 호출 제한(429)·일시 오류는 시간 예산 안에서 기다렸다 다시 부른다.

    이 재시도는 '형식이 틀려서 다시 부르기'(명세: 한 번)와 별개로, 응답 자체를 못 받은 경우에만 한다.
    """
    settings = get_settings()
    for retry in range(MAX_TRANSIENT_RETRIES + 1):
        if cancelled():
            # 같은 병렬 묶음의 다른 작업이 실패해 이 결과는 버려진다 → 토큰을 더 쓰지 않는다
            raise LLMError("같은 묶음의 다른 작업이 실패해서 중단함")
        _wait_turn(provider.name, settings.llm_min_interval)
        started = time.monotonic()
        try:
            return _call_with_slot(provider, settings, model=model, **kwargs)
        except (LLMError, LLMTimeout) as exc:
            log_event("llm_call", logging.ERROR, message=getattr(exc, "detail", "")[:400], role=role, model=model,
                      duration_ms=int((time.monotonic() - started) * 1000),
                      input_tokens=getattr(exc, "input_tokens", 0), output_tokens=getattr(exc, "output_tokens", 0),
                      error_code=exc.code)
            if not (isinstance(exc, LLMError) and exc.transient) or retry == MAX_TRANSIENT_RETRIES:
                raise
            delay = exc.retry_after if exc.retry_after is not None else TRANSIENT_BACKOFF[retry]
            left = remaining_seconds()
            if left is not None and delay > left - MIN_CALL_SECONDS:
                raise  # 기다리면 시간 예산을 넘음 → 지금 실패로 돌려준다
            time.sleep(delay)
    raise LLMError("도달할 수 없는 코드")  # pragma: no cover


def call_structured(
    task: Task,
    *,
    system: str,
    user: str,
    output_model: type[T],
    validate: Callable[[T], Outcome[V]] | None = None,
    accept_after_retries: bool = True,
    images: list[ImageInput] | None = None,
) -> Outcome[V]:
    settings = get_settings()
    tier = settings.tier(TASK_TIER[task])
    role = TASK_ROLE[task]
    try:
        provider = get_provider(tier.provider, settings)
    except LLMError as exc:
        log_event("llm_call", logging.ERROR, message=exc.detail, role=role, model=tier.model, duration_ms=0,
                  input_tokens=0, output_tokens=0, error_code=exc.code)
        raise
    schema = strict_schema(output_model)
    # 모든 호출 공통: 자료·대화·답안 안의 지시문(프롬프트 인젝션)을 따르지 않는다
    system = system.rstrip() + "\n\n" + render("input_safety")

    feedback = ""
    last: Outcome | None = None
    attempts = settings.llm_max_retries + 1
    for attempt in range(1, attempts + 1):
        prompt = user if not feedback else f"{user}\n\n[이전 응답의 문제 — 반드시 고쳐서 다시 답하세요]\n{feedback}"
        started = time.monotonic()
        try:
            result = _generate(provider, role=role, model=tier.model, task=task.value, system=system, user=prompt,
                               schema=schema, effort=tier.effort, images=images)
        except OutputTruncated as exc:
            usage().add(role, tier.model, exc.input_tokens, exc.output_tokens)   # 잘려도 비용은 나감
            if attempt == attempts:
                raise
            feedback = "- 응답이 너무 길어 중간에 잘렸습니다. 항목 수를 줄이고 문장을 짧게 써서 다시 답하세요."
            continue
        duration = int((time.monotonic() - started) * 1000)
        usage().add(role, tier.model, result.input_tokens, result.output_tokens)
        log_event("llm_call", role=role, model=tier.model, duration_ms=duration,
                  input_tokens=result.input_tokens, output_tokens=result.output_tokens)

        try:
            parsed = output_model.model_validate(json.loads(result.text))
        except (json.JSONDecodeError, ValidationError) as exc:
            feedback = f"- JSON 형식 또는 허용되지 않은 값: {str(exc)[:600]}"
            log_event("llm_invalid_output", logging.WARNING, message=f"형식 오류 (시도 {attempt}/{attempts})",
                      role=role, model=tier.model, **_raw(result.text))
            continue

        token = _last_attempt.set(attempt == attempts)
        try:
            outcome: Outcome = validate(parsed) if validate else Outcome(parsed)
        finally:
            _last_attempt.reset(token)
        last = outcome
        if not outcome.fatal:
            return outcome
        feedback = "\n".join(f"- {msg}" for msg in outcome.fatal[:15])
        log_event("llm_invalid_output", logging.WARNING,
                  message=f"근거 검증 실패 {len(outcome.fatal)}건 (시도 {attempt}/{attempts})",
                  role=role, model=tier.model, **_raw(result.text))
        # 왜 다시 시도했는지 운영 로그에서 바로 보이게 WARNING. 실패 이유에 선생님 문장 인용이 들어가므로
        # raw_output과 같은 스위치(LOG_RAW_OUTPUT)를 따른다 — false면 찍지 않는다 (건수는 위 message에 있음)
        if get_settings().log_raw_output:
            log_event("llm_grounding_detail", logging.WARNING, role=role,
                      problems=[msg[:300] for msg in outcome.fatal[:15]])

    if last is not None and accept_after_retries:
        # 근거가 확인되지 않은 항목은 validate가 이미 뺐다 → 남은 부분만 쓴다
        last.warnings.extend(last.fatal)
        last.fatal = []
        return last
    raise OutputRejected(f"[{task.value}] {attempts}번 모두 검증 실패")
