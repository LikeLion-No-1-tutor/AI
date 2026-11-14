"""로그 규칙: 표준 출력에 JSON 한 줄씩.

공통 필드: ts(한국 시간, 밀리초), level, service("ai"), env, request_id, event, message(선택)
남기지 않는 것: 원본 자료, 사용자가 쓴 내용, AI가 만든 내용, 프롬프트, 인증 정보
예외: llm_invalid_output의 raw_output(앞 2,000자), llm_grounding_detail의 problems(검증 실패 이유 — 선생님 문장 인용 포함).
      둘 다 LOG_RAW_OUTPUT=false면 남기지 않는다. DEBUG는 local에서만 찍힘.
"""
from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from .context import request_id_var

KST = ZoneInfo("Asia/Seoul")
_logger = logging.getLogger("teachinglog")
_env = "local"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, KST).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "service": "ai",
            "env": _env,
            "request_id": getattr(record, "request_id", None) or request_id_var.get(),
            "event": getattr(record, "event", "log"),
        }
        fields = getattr(record, "fields", None)
        if fields:
            data.update(fields)
        message = record.getMessage()
        if message:
            data["message"] = message
        return json.dumps(data, ensure_ascii=False, default=str)


class _StdoutHandler(logging.StreamHandler):
    """항상 현재의 sys.stdout에 쓴다 (stdout이 바뀌어도 안전)."""

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stdout
        super().emit(record)


def setup_logging(env: str) -> None:
    global _env
    _env = env
    handler = _StdoutHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    _logger.handlers = [handler]
    _logger.propagate = False
    _logger.setLevel(logging.DEBUG if env == "local" else logging.INFO)
    # 외부 라이브러리 로그는 요청 URL 등이 섞이므로 경고 이상만
    for name in ("httpx", "httpx2", "httpcore", "anthropic", "google", "google_genai", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


def log_event(event: str, level: int = logging.INFO, message: str = "", **fields: Any) -> None:
    _logger.log(level, message, extra={"event": event, "fields": fields})


def log_exception(exc: BaseException) -> None:
    # 기본 traceback에는 변수 값이 들어가지 않는다
    stack = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    log_event("exception", logging.ERROR, exception_type=type(exc).__name__, stack_trace=stack)
