"""AI-3 용어 판정 + 손 들기, AI-4 손 든 표시의 질문 만들기.

AI-3은 LLM을 두 번 '따로, 동시에' 부른다.
  - [심화, judge] 방금 보낸 메시지 하나로 용어 상태 판정 (todo → mentioned → explained, 올라가기만)
      · explained·mentioned는 메시지(답변이면 질문 포함) 원문 인용이 코드 검증을 통과해야 인정
      · 용어 이름이 메시지에 그대로 있으면 코드가 최소 mentioned로 올림 (LLM이 놓쳐도)
      · REQUIRE_TERM_NAME_FOR_EXPLAINED=true면 이름이 메시지(답변이면 질문 포함)에 없을 때 explained를 주지 않음
  - [일반, student] 학생 AI가 손을 들지 판단. 학생 프롬프트에는 judge 데이터(용어 정의)를 절대 넣지 않음.
      · 손 들기에 실패해도 용어 판정은 돌려준다 (hand_raise: null)
      · 설명이 끝난 때(손 들 때)에만 부른다. 설명 도중이면 LLM을 부르지 않고 hand_raise: null (hand_raise_point)
AI-4: [일반, student] 버튼을 누른 시점까지의 대화 + 손 든 이유로 질문 하나.
      · 질문에 수업 대화·페르소나 '원래 아는 것'에 없는 전문 용어가 있으면 이유를 붙여 다시 부른다.
        마지막 시도에서도 남으면 질문은 돌려주고 로그만 남긴다 (의성어 등 오탐으로 버튼이 실패하지 않게).
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor

from ..config import get_settings
from ..context import submit
from ..errors import LLMError, LLMTimeout
from ..grounding import mentions_term, normalize, quote_in_any, same_text, unheard_terms
from ..llm.router import Outcome, OutputRejected, Task, call_structured, is_last_attempt
from ..logging_setup import log_event, log_exception
from ..prompts import render
from ..schemas.api import AnalyzeRequest, HandRaise, JudgePart, QuestionRequest, StudentPart, TermUpdate
from ..schemas.enums import TERM_RANK, MessageKind, Sender, TermStatus
from ..schemas.llm_outputs import HandRaiseOutput, LLMTermJudgment, QuestionMakeOutput, TermJudgeOutput
from .common import index_messages, rules, student_system

# ───────────────────────── 용어 판정 (judge) ─────────────────────────


def judge_terms(judge: JudgePart) -> list[TermUpdate]:
    terms = [t for t in judge.terms if t.status != TermStatus.EXPLAINED]
    content = judge.message.content.strip()
    if not terms or not content:
        return []
    by_id = {t.term_id: t for t in terms}
    is_answer = judge.message.kind == MessageKind.ANSWER and bool(judge.reply_to_question)
    sources = [content] + ([judge.reply_to_question] if is_answer else [])
    name_sources = " ".join(sources)
    require_name = get_settings().require_name_for_explained

    def validate(out: TermJudgeOutput) -> Outcome[dict[int, TermStatus]]:
        fatal: list[str] = []
        result: dict[int, TermStatus] = {}
        for j in out.judgments:
            if j.term_id not in by_id:
                fatal.append(f"term_id {j.term_id}는 판정할 용어가 아닙니다.")
                continue
            if j.term_id in result:
                continue
            if j.status == TermStatus.TODO:
                result[j.term_id] = TermStatus.TODO
                continue
            if not quote_in_any(j.evidence_quote, sources):
                fatal.append(f"term_id {j.term_id}: {j.status.value}의 evidence_quote가 메시지 원문에 없습니다.")
                result[j.term_id] = TermStatus.TODO  # 근거 없는 판정은 올리지 않는다 (보수적)
                continue
            result[j.term_id] = j.status
        missing = [t for t in by_id if t not in result]
        if missing:
            fatal.append(f"판정이 빠진 term_id: {missing}")
        return Outcome(result, fatal=fatal)

    term_block = "\n".join(f"- term_id {t.term_id} | {t.name} (현재 {t.status.value}): {t.definition}" for t in terms)
    question_block = f"[학생 질문]\n{judge.reply_to_question}\n\n" if is_answer else ""
    kind_label = "답변(answer)" if judge.message.kind == MessageKind.ANSWER else "설명(explanation)"
    user = (f"[판정할 용어와 정의]\n{term_block}\n\n{question_block}"
            f"[사용자 메시지 — {kind_label}]\n{content}")
    judged = call_structured(Task.TERM_JUDGE, system=render("term_judge", rules=rules()), user=user,
                             output_model=TermJudgeOutput, validate=validate).value

    updates: list[TermUpdate] = []
    for t in terms:
        new = judged.get(t.term_id, TermStatus.TODO)
        # 이름은 괄호 안팎 표기를 모두 인정한다 ('순환 신경망(RNN)'이면 '순환 신경망'이나 'RNN')
        if require_name and new == TermStatus.EXPLAINED and not mentions_term(t.name, name_sources):
            new = TermStatus.TODO        # 이름을 말하지 않고 내용만 설명 → 체크하지 않음 (팀 규칙 A안)
        if new == TermStatus.TODO and mentions_term(t.name, content):
            new = TermStatus.MENTIONED   # 이름을 그대로 말했으면 최소 '언급'
        if TERM_RANK[new] > TERM_RANK[t.status]:
            updates.append(TermUpdate(term_id=t.term_id, status=new))
    log_event("term_updates", logging.DEBUG, updates=[u.model_dump(mode="json") for u in updates])
    return updates


# ───────────────────────── 손 들 때 (코드, LLM 호출 없음) ─────────────────────────
#
# 메시지마다 손 들기 LLM을 부르면 (1) 선생님 말이 안 끝났는데 질문하고 (2) 토큰이 많이 든다.
# 그래서 '설명이 한 단락 끝났다'고 볼 수 있을 때만 부른다. 서버는 상태를 저장하지 않으므로, 매 요청마다
# 대화 기록 전체를 같은 규칙으로 처음부터 다시 훑어서 '지금 메시지가 손 들 때인지'를 정한다.
#
# 손 들 때 (설명 중이 아니면서 하나라도 해당):
#   topic   새 용어로 넘어감 — 체크리스트 용어 이름이 새로 나옴
#   handoff 학생에게 넘기는 말 — "알겠지?", "질문 있어?", "여기까지", "정리하면", "다음으로" …
#   answer  학생 질문에 답함
# 손 들지 않음: 짧은 말("응", "ㅇㅋ"), 설명이 이어지는 중(":", ",", "그리고", "예를 들어" 등으로 끝남),
#   그 밖의 설명 (같은 주제를 이어서 설명하는 중이면 끝날 때까지 기다린다 — 팀 결정: 위 세 시점에만 판단)

SHORT_CHARS = 8   # 정규화 후 이보다 짧은 메시지는 맞장구·짧은 대답으로 보고 손 들 때로 치지 않음

# 설명이 이어지는 중: 이어 주는 말이나 기호로 끝나거나, 문장이 연결어미(-고/-서/-면/-는데/-록 …)에서 끊긴 메시지.
# 예: "종류는 세 가지야:", "그리고", "곱하면 곱할수록", "1보다 작아서"
_CONTINUES = re.compile(
    r"(그리고|그래서|그런데|근데|또|예를\s*들[어면]|예컨대|먼저|우선|첫째|첫\s*번째|다음과\s*같|"
    r"[:：,，→\-…]|\.\.\.?|"
    r"[가-힣](고|서|면|며|록|는데|은데|지만|니까|려고|도록|면서|거나|든지|는지|다가|자마자))\s*$")
# '-고/-서/-면'으로 끝나지만 연결어미가 아니라 명사인 낱말 (이걸로 끝나면 설명이 끝난 것으로 본다)
_NOUN_ENDINGS = (
    "참고", "최고", "광고", "경고", "사고", "재고", "창고", "원고", "보고", "신고",
    "순서", "문서", "교과서", "설명서", "계산서", "보고서", "증명서", "단서", "도서", "독서", "정서", "부서", "예시서",
    "화면", "측면", "표면", "평면", "단면", "전면", "이면", "내면", "장면", "방면", "지면", "곡면", "정면", "후면",
)


def _continues(text: str) -> bool:
    """설명이 이어지는 중인가 (끝이 연결어미·접속어·기호). '순서', '화면' 같은 명사로 끝나면 아님."""
    if not _CONTINUES.search(text):
        return False
    tail = text.rstrip()
    return not any(tail.endswith(noun) for noun in _NOUN_ENDINGS)

# 학생에게 차례를 넘기는 말 (반말·존댓말·높임말 모두). 메시지 끝 40자에서 찾는다.
_HANDOFF_END = re.compile(
    r"(알겠[지어니죠나습]|아시겠|이해(가\s*)?(돼|됐|되[니나지죠셨시]|했|하겠|하셨|하시|갔|가\s*(가|되|돼|됐))|"
    r"질문\s*(있|해|하)|궁금한\s*(거|것|점|게)|궁금하신|모르겠는\s*(거|것|게)|모르시겠|"
    r"여기까지|이상이(야|에요|었|입니다)|끝[이났!.]|어때|어떠세요|어떠신가요|괜찮[아으죠지]|됐지|됐나요|됐죠)")
# 주제를 넘기는 말. "다음 시점으로 넘어가"처럼 설명 속 말에 걸리지 않게 권유·예고 꼴만 본다
_HANDOFF_ANY = re.compile(r"(정리하면|요약하면|정리해\s*보면|다음\s*(내용|주제|거|걸)|넘어가(자|볼게|볼까|보자|겠|요)|이제\s*다음)")

HAND_HINTS = {
    "topic": "선생님이 다음 내용으로 넘어가려 한다. 넘어가기 전에, 방금 끝난 설명과 방금 들은 말에서 꼭 물어야 할 막힘이 있는지 본다.",
    "handoff": "선생님이 설명을 한 단락 마치고 학생에게 차례를 넘겼다. 지금까지 들은 설명에서 막힌 점이 있는지 본다.",
    "answer": "선생님이 학생 질문에 답했다. 답을 듣고도 여전히 막힌 점이 남았는지 본다.",
}


def _names_in(text: str, term_names: list[str]) -> set[str]:
    return {n for n in term_names if mentions_term(n, text)}


def hand_raise_point(messages: list, term_names: list[str], persona_code: str) -> str | None:
    """마지막 선생님 메시지가 '손 들 때'면 그 종류(topic/handoff/answer), 아니면 None.

    persona_code는 페르소나별 규칙을 둘 때를 위해 받는다 (지금은 모두 같은 규칙).
    """
    teacher = [m for m in messages if m.sender == Sender.TEACHER and m.content.strip()]
    if not teacher or messages[-1] is not teacher[-1]:
        return None   # 마지막 메시지가 선생님 말이 아니면 (방금 들은 말이 없으면) 손 들 때가 아님
    topic: set[str] = set()
    taught = False   # 앞에서 설명을 들은 적이 있는지 (첫 설명은 '넘어감'이 아님)
    point: str | None = None
    for m in teacher:
        text = m.content.strip()
        point = None
        names = _names_in(text, term_names)
        if len(normalize(text)) < SHORT_CHARS:
            continue   # 맞장구·짧은 대답: 주제로도 치지 않음
        if not _continues(text):
            if m.kind == MessageKind.ANSWER:
                point = "answer"
            elif names and taught and not names <= topic:
                point = "topic"
            elif _HANDOFF_END.search(text[-40:]) or _HANDOFF_ANY.search(text):
                point = "handoff"
        if names:
            topic = names
        taught = True
    return point


# ───────────────────────── 손 들기 (student) ─────────────────────────


def decide_hand_raise(student: StudentPart, point: str = "handoff") -> HandRaise | None:
    limit = get_settings().max_raised_hands   # 0 = 제한 없음 (기능명세서 기본)
    if limit and len(student.raised_reasons) >= limit:
        return None
    indexed = index_messages(student.messages)
    if not indexed.teacher:
        return None
    existing = list(student.raised_reasons) + indexed.student_questions

    def validate(out: HandRaiseOutput) -> Outcome[HandRaise | None]:
        if not out.raise_hand:
            return Outcome(None)
        reason = (out.reason or "").strip()
        if not reason:
            return Outcome(None, fatal=["raise_hand=true이면 reason이 필요합니다."])
        if any(same_text(reason, e) for e in existing):
            return Outcome(None)  # 같은 이유로 또 손 들지 않음
        return Outcome(HandRaise(reason=reason))

    reasons = "\n".join(f"- {r}" for r in student.raised_reasons) or "(없음)"
    asked = "\n".join(f"- {q}" for q in indexed.student_questions) or "(없음)"
    system = student_system(student.persona, render("hand_raise"))
    user = (f"[지금까지 수업 대화 — 마지막 선생님 메시지가 방금 들은 말]\n{indexed.format(student.persona.name)}\n\n"
            f"[지금 상황]\n{HAND_HINTS.get(point, HAND_HINTS['handoff'])}\n\n"
            f"[이미 손 든 이유 (아직 질문하지 않음)]\n{reasons}\n\n"
            f"[이미 한 질문]\n{asked}")
    return call_structured(Task.HAND_RAISE, system=system, user=user, output_model=HandRaiseOutput,
                           validate=validate, accept_after_retries=False).value


def analyze(req: AnalyzeRequest) -> tuple[list[TermUpdate], HandRaise | None]:
    point = hand_raise_point(req.student.messages, [t.name for t in req.judge.terms], req.student.persona.code)
    if point is None:
        # 설명 도중 → 손 들기 LLM을 부르지 않는다 (말 끊지 않기 + 토큰 절약)
        log_event("hand_raise_not_now", logging.DEBUG)
        return judge_terms(req.judge), None
    with ThreadPoolExecutor(max_workers=2) as pool:
        judge_f = submit(pool, judge_terms, req.judge)
        hand_f = submit(pool, decide_hand_raise, req.student, point)
        updates = judge_f.result()            # 판정 실패는 요청 실패
        try:
            hand = hand_f.result()
        except (LLMError, LLMTimeout, OutputRejected) as exc:
            log_event("hand_raise_skipped", logging.DEBUG, error_code=getattr(exc, "code", "LLM_ERROR"))
            hand = None
        except Exception as exc:  # noqa: BLE001 — 손 들기는 부가 기능: 코드 오류여도 판정 결과는 돌려준다
            log_exception(exc)    # 명세 로그 이벤트 'exception' (코드 오류라 prod에서도 남긴다)
            hand = None
    return updates, hand


# ───────────────────────── AI-4 질문 만들기 ─────────────────────────


# 아래첨자 기호 표기 (h_t, x_{t-1}, W_hh …). 선생님이 쓴 표기만 쓸 수 있다 (h_t를 듣고 h_{t-1}을 만들면 안 됨)
_NOTATION = re.compile(r"[A-Za-z][A-Za-z0-9]*_(?:\{[^{}\s]+\}|[A-Za-z0-9+\-]+)")


def make_question(req: QuestionRequest) -> str:
    indexed = index_messages(req.messages)
    asked = indexed.student_questions
    heard = [m.content for _, m in indexed.items] + [req.persona.system_prompt]
    heard_norm = normalize(" ".join(heard))

    def validate(out: QuestionMakeOutput) -> Outcome[str]:
        text = out.question.strip()
        if not text:
            return Outcome("", fatal=["question이 비어 있습니다."])
        if any(same_text(text, a) for a in asked):
            return Outcome(text, fatal=["이미 한 질문과 같습니다. 다른 질문을 하세요."])
        if not is_last_attempt():
            # 한 번에 하나만 묻는다: 물음표가 둘 이상이면 선생님이 무엇에 답할지 흐려진다
            if text.count("?") + text.count("？") > 1:
                return Outcome(text, fatal=["물음표가 두 개 이상입니다. 가장 궁금한 질문 하나만, 물음표 하나로 쓰세요."])
            notation = [n for n in _NOTATION.findall(text) if normalize(n) not in heard_norm]
            if notation:
                return Outcome(text, fatal=[f"선생님이 쓰지 않은 기호 표기({', '.join(notation)})가 있습니다. "
                                            "선생님이 쓴 표기나 말로만 질문하세요."])
        unheard = unheard_terms(text, heard)
        # 다시 부르는 건 영문 용어(LSTM 등)만. 한글은 글자 비교라 '줄어드는'·'끊어야' 같은 평범한 말도 걸려서
        # 다시 부를 근거로 쓰지 않고 기록만 남긴다.
        latin = [w for w in unheard if w.isascii()]
        if latin and not is_last_attempt():
            return Outcome(text, fatal=[f"질문에 수업에서 나오지 않은 말({', '.join(latin)})이 있습니다. "
                                        "선생님이 쓴 말과 평소 쓰는 쉬운 말로만 질문하세요."])
        if unheard:
            log_event("question_unheard_words", logging.DEBUG, count=len(unheard), latin=len(latin))
        return Outcome(text)

    system = student_system(req.persona, render("question_make"))
    user = (f"[지금까지 수업 대화]\n{indexed.format(req.persona.name)}\n\n"
            f"[손 든 이유]\n{req.reason.strip()}")
    return call_structured(Task.QUESTION_MAKE, system=system, user=user, output_model=QuestionMakeOutput,
                           validate=validate, accept_after_retries=False).value
