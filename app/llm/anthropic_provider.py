"""Claude (Anthropic) 공급자.

- Haiku 5.5는 temperature/top_p/top_k를 받지 않으므로 보내지 않는다.
- 형식은 structured outputs(output_config.format = json_schema)로 강제한다.
- 깊이는 output_config.effort (low/medium/high). 빈 값이면 모델 기본값.
- SDK 자체 재시도는 끄고(max_retries=0), 재시도와 시간 예산은 router가 관리한다.
"""
from __future__ import annotations

import base64
from typing import Any

import anthropic

from ..errors import LLMError, LLMTimeout, OutputTruncated
from .base import ImageInput, LLMProvider, LLMResult


def create(settings) -> "AnthropicProvider":
    return AnthropicProvider(settings.anthropic_api_key, settings.llm_max_output_tokens)


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str, max_tokens: int):
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY가 설정되지 않았습니다.")
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=0)
        self._max_tokens = max_tokens

    def generate_json(self, *, task: str, model: str, system: str, user: str, schema: dict[str, Any],
                      timeout: float, effort: str = "", images: list[ImageInput] | None = None) -> LLMResult:
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if effort:
            output_config["effort"] = effort
        content: list[dict[str, Any]] = [
            {"type": "image", "source": {"type": "base64", "media_type": img.mime_type,
                                         "data": base64.standard_b64encode(img.data).decode()}}
            for img in images or []
        ]
        content.append({"type": "text", "text": user})
        try:
            response = self._client.with_options(timeout=timeout, max_retries=0).messages.create(
                model=model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": content}],
                output_config=output_config,
            )
        except anthropic.APITimeoutError as exc:
            raise LLMTimeout(str(exc)) from exc
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as exc:
            retry_after = None
            response = getattr(exc, "response", None)
            if response is not None and response.headers.get("retry-after", "").replace(".", "", 1).isdigit():
                retry_after = float(response.headers["retry-after"])
            raise LLMError(f"Claude 일시 오류: {type(exc).__name__}: {str(exc)[:200]}", transient=True,
                           retry_after=retry_after) from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude 요청 오류 ({exc.status_code}): {str(exc.message)[:300]}") from exc

        if response.stop_reason == "max_tokens":
            # 같은 요청으로 기다렸다 재시도하지 않는다. router가 '더 짧게'를 붙여 한 번 다시 부른다.
            raise OutputTruncated(f"Claude 응답이 max_tokens({self._max_tokens})에서 잘렸습니다.",
                                  input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
        if response.stop_reason == "refusal":
            raise LLMError("Claude가 응답을 거절했습니다.")
        text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        if not text:
            raise LLMError("Claude 응답에 텍스트가 없습니다.", transient=True)
        return LLMResult(text=text, input_tokens=response.usage.input_tokens,
                         output_tokens=response.usage.output_tokens)
