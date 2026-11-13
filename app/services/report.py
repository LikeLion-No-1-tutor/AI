"""AI-9 성장보고서 작성 (역할: report, 일반).

AI는 문장 두 가지만 만든다. 점수·놓친 개념·점수 비교는 Django가 DB로 계산한다.
- strengths: 근거가 선생님 메시지 원문 인용(코드 검증) 또는 학생 AI가 맞힌 문제 번호인 것만 남긴다.
- review_recommendations: 대상 = explained가 아닌 용어 + 관련 오답이 있는 용어.
  대상이 아닌 term_id는 버리고, 빠진 대상은 코드가 용어 정의로 기본 문장을 채운다.
  정의와 어긋나게 설명한 용어는 "~는 맞게 설명했지만, '~'라고 설명한 부분은 자료와 달라요" 꼴로 쓰고,
  어긋난 부분(wrong_quote)·맞은 부분(right_quote)이 선생님 메시지 원문에 있는지 코드가 확인한다.
"""
from __future__ import annotations

from ..errors import validation_error
from ..grounding import quote_in_any
from ..llm.router import Outcome, Task, call_structured
from ..prompts import render
from ..schemas.api import Recommendation, ReportRequest
from ..schemas.enums import TermStatus
from ..schemas.llm_outputs import LLMRecommendation, LLMStrength, ReportOutput
from .common import index_messages, rules

MAX_STRENGTHS = 3


def review_targets(req: ReportRequest) -> list[int]:
    wrong_terms = {q.term_id for q in req.exam.questions if q.is_correct is False and q.term_id is not None}
    return [t.term_id for t in req.terms if t.status != TermStatus.EXPLAINED or t.term_id in wrong_terms]


def fallback_recommendation(name: str, definition: str) -> str:
    return f"{name}: {definition} 이 내용을 자료로 다시 확인하고, 자기 말로 설명해 보세요."


_DIFFERS = "자료와 달라"


def recommendation_problem(r: LLMRecommendation, teacher_texts: list[str]) -> str | None:
    """'맞게 설명했지만 ~라고 설명한 부분은 자료와 달라요' 문장의 근거를 확인한다. 문제가 없으면 None."""
    if r.wrong_quote and r.wrong_quote.strip():
        if not quote_in_any(r.wrong_quote, teacher_texts):
            return "wrong_quote가 선생님 메시지에 글자 그대로 없습니다. 어긋난 부분을 원문 그대로 복사하세요."
        if r.right_quote and r.right_quote.strip() and not quote_in_any(r.right_quote, teacher_texts):
            return "right_quote가 선생님 메시지에 글자 그대로 없습니다. 맞는 부분을 원문 그대로 복사하거나 null로 두세요."
        if _DIFFERS not in r.text:
            return "wrong_quote가 있으면 text에 \"'~'라고 설명한 부분은 자료와 달라요\"를 넣으세요."
    elif _DIFFERS in r.text:
        return "'자료와 달라요'라고 쓰려면 어긋난 부분을 wrong_quote에 선생님 메시지 원문 그대로 복사하세요."
    return None


def check_exam_numbers(req: ReportRequest) -> None:
    """Django가 보낸 점수 숫자가 문항 목록과 맞는지 (틀리면 잘못된 근거로 보고서를 쓰게 되므로 422)."""
    exam = req.exam
    numbers = [q.number for q in exam.questions]
    if len(numbers) != len(set(numbers)):
        raise validation_error("exam.questions의 number가 중복되었어요")
    if exam.total != len(exam.questions):
        raise validation_error(f"exam.total({exam.total})이 문항 수({len(exam.questions)})와 달라요")
    if not 0 <= exam.correct <= exam.total:
        raise validation_error("exam.correct는 0 이상 total 이하여야 해요")
    # 채점 안 된 문항(is_correct=null)이 있으면 correct는 '맞음 수' ~ '맞음 수 + 미채점 수' 사이여야 한다
    right = sum(1 for q in exam.questions if q.is_correct is True)
    unmarked = sum(1 for q in exam.questions if q.is_correct is None)
    if not right <= exam.correct <= right + unmarked:
        expected = f"{right}" if not unmarked else f"{right}~{right + unmarked} (미채점 {unmarked}문항 포함)"
        raise validation_error(f"exam.correct({exam.correct})가 맞힌 문항 수({expected})와 맞지 않아요")


def write_report(req: ReportRequest) -> tuple[list[str], list[Recommendation]]:
    check_exam_numbers(req)
    indexed = index_messages(req.messages)
    targets = review_targets(req)
    term_map = {t.term_id: t for t in req.terms}
    correct_numbers = {q.number for q in req.exam.questions if q.is_correct}

    def validate(out: ReportOutput) -> Outcome[ReportOutput]:
        fatal: list[str] = []
        strengths: list[LLMStrength] = []
        for s in out.strengths:
            if not s.text.strip():
                continue
            refs = [r for r in s.message_refs if r in indexed.teacher]
            quote_ok = bool(refs) and quote_in_any(s.quote, [indexed.teacher[r] for r in refs])
            questions_ok = bool(s.question_numbers) and set(s.question_numbers) <= correct_numbers
            if quote_ok or questions_ok:
                strengths.append(s)
            else:
                fatal.append(f"잘한 점 '{s.text[:20]}…': 근거(선생님 메시지 인용 또는 맞힌 문제 번호)가 확인되지 않습니다.")
        recs = [r for r in out.review_recommendations if r.term_id in targets and r.text.strip()]
        if len(recs) != len(out.review_recommendations):
            fatal.append(f"review_recommendations에는 복습 대상 term_id {targets}만 쓸 수 있습니다.")
        teacher_texts = list(indexed.teacher.values())
        checked: list[LLMRecommendation] = []
        for r in recs:
            problem = recommendation_problem(r, teacher_texts)
            if problem:
                # 근거가 확인되지 않은 '틀린 설명' 문장은 버린다 → 빠진 대상은 아래에서 기본 문장으로 채움
                fatal.append(f"term_id {r.term_id} 복습 추천: {problem}")
            else:
                checked.append(r)
        recs = checked
        return Outcome(out.model_copy(update={"strengths": strengths[:MAX_STRENGTHS], "review_recommendations": recs}),
                       fatal=fatal)

    term_lines = "\n".join(f"- term_id {t.term_id} | {t.name} [{t.status.value}]: {t.definition}" for t in req.terms)
    target_lines = "\n".join(f"- term_id {i} | {term_map[i].name}" for i in targets) or "(없음)"
    q_lines = "\n".join(
        f"- {q.number}번 ({'맞음' if q.is_correct else '틀림'}, term_id {q.term_id}): {q.content} / 학생 답: {q.response}"
        for q in req.exam.questions
    ) or "(문제 없음)"
    system = render("report", student=f"{req.persona.name}({req.persona.grade_label})", rules=rules())
    user = (f"[수업 주제] {req.title}\n\n[용어와 상태]\n{term_lines}\n\n[복습 대상 용어]\n{target_lines}\n\n"
            f"[학생 AI 평가] {req.exam.correct}/{req.exam.total} 맞힘\n{q_lines}\n\n"
            f"[수업 대화]\n{indexed.format(req.persona.name)}")

    if indexed.teacher or correct_numbers or targets:
        out = call_structured(Task.REPORT, system=system, user=user, output_model=ReportOutput, validate=validate).value
    else:
        out = ReportOutput(strengths=[], review_recommendations=[])

    by_term = {}
    for r in out.review_recommendations:
        by_term.setdefault(r.term_id, r.text.strip())
    recommendations = [
        Recommendation(term_id=i, text=by_term.get(i) or fallback_recommendation(term_map[i].name, term_map[i].definition))
        for i in targets
    ]
    return [s.text.strip() for s in out.strengths], recommendations
