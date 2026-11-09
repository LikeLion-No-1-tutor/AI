"""AI 서버 에러. 응답 형식: {"error": {"code", "message", "request_id"}}

API 명세의 code만 쓴다.
| HTTP | code                   |
| 401  | INVALID_INTERNAL_TOKEN |
| 422  | VALIDATION_ERROR       |
| 422  | UNREADABLE_MATERIAL    |  (AI-1, AI-2만)
| 502  | LLM_ERROR              |
| 504  | LLM_TIMEOUT            |
"""
from __future__ import annotations


class AppError(Exception):
    def __init__(self, code: str, message: str, status_code: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.detail = ""   # 원인 설명 (응답에는 넣지 않고 로그 message에만)


def invalid_token() -> AppError:
    return AppError("INVALID_INTERNAL_TOKEN", "내부 인증 토큰이 올바르지 않아요.", 401)


def validation_error(detail: str = "") -> AppError:
    err = AppError("VALIDATION_ERROR", "요청 형식이 올바르지 않아요.", 422)
    err.detail = detail
    return err


def unreadable(detail: str = "") -> AppError:
    err = AppError("UNREADABLE_MATERIAL", "자료에서 내용을 읽을 수 없어요.", 422)
    err.detail = detail
    return err


class LLMError(AppError):
    """LLM 호출 실패. transient=True면 시간이 남아 있을 때 한 번 더 시도할 수 있음."""

    def __init__(self, detail: str = "", transient: bool = False, retry_after: float | None = None):
        super().__init__("LLM_ERROR", "LLM 응답을 처리하지 못했어요.", 502)
        self.detail = detail
        self.transient = transient
        self.retry_after = retry_after   # 호출 제한(429) 등에서 공급자가 알려준 대기 시간(초)


class OutputTruncated(LLMError):
    """응답이 max_tokens에서 잘림. 같은 요청을 그대로 다시 보내면 또 잘리므로, '더 짧게'를 붙여 다시 부른다.

    잘린 응답도 토큰 비용은 나가므로 사용량을 함께 들고 온다 (usage[]에 기록).
    """

    def __init__(self, detail: str = "", input_tokens: int = 0, output_tokens: int = 0):
        super().__init__(detail)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class LLMTimeout(AppError):
    def __init__(self, detail: str = ""):
        super().__init__("LLM_TIMEOUT", "LLM 응답이 시간 안에 오지 않았어요.", 504)
        self.detail = detail
