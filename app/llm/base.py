"""모든 LLM 공급자가 따르는 인터페이스."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ImageInput:
    data: bytes
    mime_type: str   # image/png, image/jpeg, image/webp, image/gif


@dataclass(frozen=True)
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def generate_json(
        self,
        *,
        task: str,
        model: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        timeout: float,
        effort: str = "",
        images: list[ImageInput] | None = None,
    ) -> LLMResult:
        """JSON 문자열과 토큰 사용량을 돌려준다.

        실패하면 app.errors.LLMError, 시간 초과면 app.errors.LLMTimeout을 던진다.
        """
