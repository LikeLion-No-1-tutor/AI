"""프롬프트 파일 로더. 파일 안의 {{이름}} 자리를 값으로 바꾼다 (JSON 중괄호와 충돌하지 않게)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_DIR = Path(__file__).parent


@lru_cache
def _read(name: str) -> str:
    return (_DIR / f"{name}.md").read_text(encoding="utf-8")


def render(name: str, **values: str) -> str:
    text = _read(name)
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    if "{{" in text:
        start = text.index("{{")
        raise KeyError(f"프롬프트 '{name}'에 채우지 않은 자리가 있습니다: {text[start:start + 30]}")
    return text
