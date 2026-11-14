"""환경 변수 설정. 모델 구성은 app/edition.py (배포판마다 다름)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

from .edition import EDITION, EDITIONS

load_dotenv()


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    raw = _env(name)
    try:
        value = int(raw) if raw else default
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} 는 정수여야 합니다") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name}={value} 는 {minimum} 이상이어야 합니다")
    return value


def _env_float(name: str, default: float, minimum: float | None = None) -> float:
    raw = _env(name)
    try:
        value = float(raw) if raw else default
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} 는 숫자여야 합니다") from exc
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{name}={raw!r} 는 유한한 숫자여야 합니다")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name}={value} 는 {minimum} 이상이어야 합니다")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name}={raw!r} 는 true/false (1/0, yes/no, on/off) 중 하나여야 합니다")


@dataclass(frozen=True)
class TierConfig:
    provider: str
    model: str
    effort: str  # Claude 전용 (low/medium/high). 빈 문자열이면 보내지 않음 (모델 기본값)


@dataclass(frozen=True)
class Settings:
    app_env: str
    edition: str
    light: TierConfig
    deep: TierConfig
    judge: TierConfig
    anthropic_api_key: str
    ai_server_token: str
    realtime_timeout: int
    background_timeout: int
    llm_max_retries: int
    llm_max_output_tokens: int
    llm_concurrency: int
    llm_global_concurrency: int
    llm_min_interval: float
    max_pdf_pages: int
    max_upload_mb: int
    max_image_mb: int
    pdf_render_dpi: int
    pages_per_read: int
    min_terms: int
    max_terms: int
    max_raised_hands: int
    require_name_for_explained: bool
    enable_answer_verification: bool
    enable_definition_verification: bool
    enable_claim_verification: bool
    log_raw_output: bool

    def tier(self, name: str) -> TierConfig:
        return {"light": self.light, "deep": self.deep, "judge": self.judge}[name]


def _tier(edition: dict, name: str) -> TierConfig:
    default_model, default_effort = edition[name]
    suffix = name.upper()
    effort = _env(f"ANTHROPIC_EFFORT_{suffix}", default_effort) if edition["provider"] == "anthropic" else ""
    return TierConfig(provider=edition["provider"], model=_env(f"MODEL_{suffix}", default_model), effort=effort)


@lru_cache
def get_settings() -> Settings:
    edition_name = _env("TEACHINGLOG_EDITION", EDITION)
    if edition_name not in EDITIONS:
        raise ValueError(f"TEACHINGLOG_EDITION={edition_name} 는 {sorted(EDITIONS)} 중 하나여야 합니다")
    edition = EDITIONS[edition_name]
    app_env = _env("APP_ENV", "local").lower()
    if app_env not in {"local", "staging", "prod"}:
        raise ValueError("APP_ENV는 local / staging / prod 중 하나여야 합니다")
    settings = Settings(
        app_env=app_env,
        edition=edition_name,
        light=_tier(edition, "light"),
        deep=_tier(edition, "deep"),
        judge=_tier(edition, "judge"),
        anthropic_api_key=_env("ANTHROPIC_API_KEY"),
        ai_server_token=_env("AI_SERVER_TOKEN"),
        realtime_timeout=_env_int("REALTIME_LLM_TIMEOUT", 25, 5),
        background_timeout=_env_int("BACKGROUND_LLM_TIMEOUT", 170, 5),
        llm_max_retries=_env_int("LLM_MAX_RETRIES", 1, 0),
        llm_max_output_tokens=_env_int("LLM_MAX_OUTPUT_TOKENS", 16000, 256),
        llm_concurrency=_env_int("LLM_CONCURRENCY", 4, 1),
        llm_global_concurrency=_env_int("LLM_GLOBAL_CONCURRENCY", 8, 0),
        llm_min_interval=_env_float("LLM_MIN_INTERVAL_SECONDS", 0.0, 0.0),
        max_pdf_pages=_env_int("MAX_PDF_PAGES", 50, 1),
        max_upload_mb=_env_int("MAX_UPLOAD_MB", 20, 1),
        max_image_mb=_env_int("MAX_IMAGE_MB", 5, 1),
        pdf_render_dpi=_env_int("PDF_RENDER_DPI", 130, 50),
        pages_per_read=max(1, _env_int("PAGES_PER_READ", 4)),
        min_terms=_env_int("MIN_TERMS", 3, 1),
        max_terms=_env_int("MAX_TERMS", 10, 1),
        max_raised_hands=_env_int("MAX_RAISED_HANDS", 0, 0),
        require_name_for_explained=_env_bool("REQUIRE_TERM_NAME_FOR_EXPLAINED", True),
        enable_answer_verification=_env_bool("ENABLE_ANSWER_VERIFICATION", True),
        enable_definition_verification=_env_bool("ENABLE_DEFINITION_VERIFICATION", True),
        enable_claim_verification=_env_bool("ENABLE_CLAIM_VERIFICATION", True),
        # API 명세 로그 규칙: llm_invalid_output에 raw_output(앞 2,000자). false면 원문 대신 글자 수만 남김
        log_raw_output=_env_bool("LOG_RAW_OUTPUT", True),
    )
    if settings.min_terms > settings.max_terms:
        raise ValueError(f"MIN_TERMS({settings.min_terms})는 MAX_TERMS({settings.max_terms}) 이하여야 합니다")
    return settings
