"""이 서버의 모델 구성. 지금은 모든 기능이 claude-haiku-5-5 하나다 (PRD 4-3).

light / deep / judge는 나중에 일부 기능만 모델을 바꿀 수 있게 남겨 둔 설정 칸이다 (지금은 셋 다 같은 모델).
judge만 같은 모델을 effort high로 부른다 (수업 중 용어 판정 정확도).
- light: 결과를 코드로 다시 검사할 수 있거나 틀려도 피해가 작은 작업
         (자료 쪽·이미지 읽기, 손 들기, 질문 만들기, 정리노트, 학생 문제 풀이, 성장보고서 문장)
- deep : 이후 단계의 기준이 되거나 점수에 직접 들어가는 판정 (용어, 요약정리, 출제·정답 검사, 주관식 채점)
- judge: 수업 중 메시지마다 실시간으로 도는 용어 판정 (25초 안에 끝나야 함)

모델 이름은 .env의 MODEL_LIGHT / MODEL_DEEP / MODEL_JUDGE로 바꿀 수 있다.
"""

EDITION = 'claude-haiku'

EDITIONS = {
    'claude-haiku': {
        'provider': 'anthropic',
        'light': ('claude-haiku-5-5', ''),
        'deep': ('claude-haiku-5-5', ''),
        'judge': ('claude-haiku-5-5', 'high'),
    },
}
