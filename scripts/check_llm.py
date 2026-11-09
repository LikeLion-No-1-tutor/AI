"""API 키와 모델이 제대로 연결되는지 바로 확인하는 스크립트 (서버를 켜지 않아도 됨).

사용법: python scripts/check_llm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.errors import LLMError, LLMTimeout  # noqa: E402
from app.llm.json_schema import strict_schema  # noqa: E402
from app.llm.router import get_provider  # noqa: E402


class Ping(BaseModel):
    answer: str


def main() -> int:
    s = get_settings()
    provider_name = s.light.provider
    print(f"구성: {s.edition}  (APP_ENV={s.app_env})")
    if provider_name == "anthropic":
        print(f"ANTHROPIC_API_KEY: {'있음' if s.anthropic_api_key else '없음'}")
    ok = True
    checked: set[tuple[str, str]] = set()
    for name in ("light", "deep", "judge"):
        tier = s.tier(name)
        print(f"\n[{name}] {tier.provider} / {tier.model}" + (f" / effort={tier.effort}" if tier.effort else ""))
        if (tier.model, tier.effort) in checked:
            print("  (위와 같은 모델 — 건너뜀)")
            continue
        checked.add((tier.model, tier.effort))
        try:
            provider = get_provider(tier.provider, s)
            result = provider.generate_json(task="ping", model=tier.model, system="JSON으로만 답하세요.",
                                            user='answer에 "안녕"이라고 쓰세요.', schema=strict_schema(Ping),
                                            timeout=60, effort=tier.effort)
            print(f"  성공: {result.text}  (입력 {result.input_tokens}, 출력 {result.output_tokens} 토큰)")
        except (LLMError, LLMTimeout) as exc:
            ok = False
            print(f"  실패 [{exc.code}]: {exc.detail}")
    if not ok:
        print("\n자주 있는 원인:\n"
              "- 키 없음 / 401: .env의 API 키가 비었거나 틀림 (앞뒤 공백, 따옴표 확인)\n"
              "- 404 / not_found: 이 키로 쓸 수 없는 모델 → .env의 MODEL_LIGHT / MODEL_DEEP / MODEL_JUDGE 에 쓸 수 있는 모델 이름 입력\n"
              "- 400 credit / billing: 결제(크레딧) 등록 필요\n"
              "- 429: 사용량 한도 초과 (무료 등급이면 잠시 후 다시, 하루 한도면 다음 날)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
