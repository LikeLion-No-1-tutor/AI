"""페르소나.

- 실제 페르소나(system_prompt)는 Django의 personas 테이블에 있고, 요청마다 받는다.
- personas_seed.json은 AI 파트가 만든 시드 데이터다. Django가 personas 테이블을 채울 때 쓴다.
- AI 서버는 받은 system_prompt 뒤에 '학생 공통 규칙'(prompts/student_rules.md)을 항상 붙여서,
  Django 쪽 프롬프트가 바뀌어도 '가르친 것만 안다' 규칙은 지켜지게 한다.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

SEED_PATH = Path(__file__).parent / "personas_seed.json"


REACTIONS_PATH = Path(__file__).parent / "reactions.json"


@lru_cache
def seed() -> list[dict]:
    return json.loads(SEED_PATH.read_text(encoding="utf-8"))


@lru_cache
def _reactions() -> dict[str, list[str]]:
    data = json.loads(REACTIONS_PATH.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if isinstance(v, list) and v}


VOICES_PATH = Path(__file__).parent / "voices.json"


@lru_cache
def _voices() -> dict[str, dict]:
    data = json.loads(VOICES_PATH.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def voice_block(persona_code: str, name: str, part: str) -> str:
    """이 페르소나의 글투 (공통 + 이번 작업 part). part: 질문 / 노트 / 시험. 없는 페르소나면 빈 문자열.

    다른 페르소나 예시는 넣지 않는다 (섞여 보이면 말투가 평균으로 수렴한다 — 실측).
    """
    v = _voices().get(persona_code)
    if not v:
        return ""
    lines = [f"[{name}의 글투 — 이 학생이 쓰는 모든 문장에 반드시 적용. 노트·답만 봐도 {name}이(가) 쓴 것을 알 수 있어야 한다]"]
    lines += [f"- {x}" for x in v.get("글투", [])]
    parts = [part] + (["궁금"] if part == "노트" else [])
    for p in parts:
        if v.get(p):
            label = "아직 궁금한 사항" if p == "궁금" else p
            lines.append(f"[{label}]")
            lines += [f"- {x}" for x in v[p]]
    lines.append("- 글투는 말하는 방식만 바꾼다. 선생님이 말하지 않은 사실·용어·숫자는 글투를 핑계로 보태지 않는다.")
    return "\n".join(lines)


def note_check(persona_code: str) -> dict | None:
    """정리노트에서 코드로 확인할 글투 표시 {regex, min_ratio, hint}. 없으면 None."""
    return (_voices().get(persona_code) or {}).get("note_check")


def reaction_for(persona_code: str, nth: int) -> str:
    """페르소나의 고정 반응 문장. nth(0부터)번째 답변이면 목록을 차례로 돌려 쓴다 (연달아 같은 문장이 나오지 않게).

    reactions.json에 없는 페르소나(Django에서 새로 만든 것)는 '_default'를 쓴다.
    """
    table = _reactions()
    lines = table.get(persona_code) or table["_default"]
    return lines[nth % len(lines)]
