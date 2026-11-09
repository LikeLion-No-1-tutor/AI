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


@lru_cache
def seed() -> list[dict]:
    return json.loads(SEED_PATH.read_text(encoding="utf-8"))
