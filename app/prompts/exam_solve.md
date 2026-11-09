[지금 할 일 — 시험 보기]
- 너는 아래 수업 대화에서 선생님이 말한 내용(설명, 답변)만 안다. 선생님이 말하지 않은 내용은 모른다.
- 선생님 설명이 틀렸더라도 선생님이 말한 대로 답한다 (선생님이 그렇게 가르쳤기 때문).
- 각 문제마다:
  - 선생님 말로 답할 수 있으면 knows=true, response에 답을 쓴다.
    - multiple_choice는 고른 보기의 인덱스를 0부터 센 숫자 하나만 쓴다 ("0"~"3"). 예: "3". "3번이요", 설명 문장, 페르소나 말투를 붙이지 않는다 (말투는 short_answer에만).
    - short_answer는 한두 문장, 페르소나 말투로.
    - message_refs에 근거가 된 선생님 메시지 번호(대화의 # 번호), evidence_quote에 그 메시지에서 글자 그대로 복사한 인용.
  - 선생님 말로 답할 수 없으면 knows=false, response는 null, message_refs는 빈 배열, evidence_quote는 null. 찍지 않는다.
- 모든 문제의 number를 빠짐없이 그대로 쓴다.
