"""AI-6 평가·재시험 출제 (evaluator, 심화), AI-7 학생 AI 문제 풀이 (student, 일반).

AI-6
  - 출제 계획(번호별 용어·유형)을 코드가 먼저 정한다: 재시험 focus 용어 → 없으면 전체 용어(중요한 순),
    유형은 question_types를 차례로 돌림. 요약정리에 내용이 없는 용어는 계획에서 뺀다.
  - 요약정리를 줄 번호(L1…)로 보여주고, 문제마다 근거 줄 번호 + 원문 인용을 받아 코드로 검증.
  - 정답 독립 검사(다른 호출)로 정답이 근거와 맞고 하나뿐인지 확인 → 통과한 문제만 쓴다.
  - 부족한 번호는 다시 출제(최대 3라운드). 그래도 count개가 안 되면 502 LLM_ERROR (명세: 정확히 count개).
  - 객관식 보기 순서는 코드가 섞고 정답 인덱스(0부터)를 다시 계산.
AI-7
  - 학생 AI는 수업 대화만 본다. 답을 냈다면 근거 선생님 메시지 번호 + 원문 인용이 있어야 하고,
    검증을 통과하지 못하면 '모르겠어요'로 바꾼다 (아는 척 방지).
"""
from __future__ import annotations

import hashlib
import logging
import random
import re
from dataclasses import dataclass

from ..config import get_settings
from ..errors import validation_error
from ..grounding import answer_leaked, normalize, numbers_unsupported, quote_in_any, same_text
from ..llm.router import Outcome, OutputRejected, Task, call_structured
from ..logging_setup import log_event
from ..prompts import render
from ..schemas.api import ExamTerm, GeneratedQuestion, GenerateRequest, SolveAnswer, SolveRequest
from ..schemas.enums import ExamKind, QuestionType
from ..schemas.llm_outputs import AnswerVerificationOutput, ExamGenerationOutput, ExamSolveOutput, LLMQuestion
from .common import (UNKNOWN_RESPONSE, claim_verification_on, index_messages, rules, student_system,
                     teacher_context, verify_claims)
from .summary_doc import ParsedSummary, parse_summary

MAX_ROUNDS = 3
CHOICE_COUNT = 4


def normalize_choice_answer(answer: str | None, choices: list[str] | None) -> str | None:
    """객관식 답 → 보기 인덱스(0부터) 문자열. 보기 글자를 그대로 쓴 답도 인덱스로 바꾼다."""
    if answer is None or not choices:
        return None
    a = answer.strip().strip(".)번 ")
    if a.isdigit() and 0 <= int(a) < len(choices):
        return str(int(a))
    for i, c in enumerate(choices):
        if same_text(a, c):
            return str(i)
    # 페르소나 말투로 문장째 답한 경우 ("3번이요. 이전 시점의…", "'3'번이요!"): 보기 번호가 정확히 하나만 있을 때만 인정.
    # 숫자가 둘 이상이거나(보기 글자 속 숫자 포함) 범위 밖이면 None → 다시 부른다.
    digits = set(_STANDALONE_DIGIT.findall(a))
    if len(digits) == 1:
        n = int(next(iter(digits)))
        if 0 <= n < len(choices):
            return str(n)
    return None


_STANDALONE_DIGIT = re.compile(r"(?<![\d.])(\d+)(?![\d.]\d)")


@dataclass(frozen=True)
class Slot:
    index: int                 # 0부터
    term: ExamTerm | None
    question_type: QuestionType


def plan_slots(req: GenerateRequest, summary: ParsedSummary) -> list[Slot]:
    usable = [t for t in req.terms if summary.lines_for(t.name)]
    by_id = {t.term_id: t for t in usable}
    wanted = list(dict.fromkeys(req.focus_term_ids))
    known_ids = {t.term_id for t in req.terms}
    unknown = [i for i in wanted if i not in known_ids]
    if unknown:
        raise validation_error(f"focus_term_ids {unknown}가 terms에 없어요")
    focus = [by_id[i] for i in wanted if i in by_id]
    if wanted and not focus:
        # 조용히 전체 용어로 내면 재시험이 엉뚱한 용어로 나간다
        raise validation_error(f"focus_term_ids {wanted}의 용어가 요약정리에 없어서 재시험 문제를 낼 수 없어요")
    if len(focus) < len(wanted):
        log_event("exam_focus_terms_skipped", logging.DEBUG,
                  message="요약정리에 내용이 없는 재시험 용어는 빼고 출제", requested=len(wanted), used=len(focus))
    targets: list[ExamTerm | None] = list(focus or usable) or [None]
    types = req.question_types
    return [Slot(i, targets[i % len(targets)], types[i % len(types)]) for i in range(req.count)]


def _shuffle(q: LLMQuestion) -> LLMQuestion:
    if q.question_type != QuestionType.MULTIPLE_CHOICE or not q.choices:
        return q
    correct = q.choices[int(q.correct_answer)]
    seed = int(hashlib.sha256(q.content.encode()).hexdigest()[:8], 16)
    choices = list(q.choices)
    random.Random(seed).shuffle(choices)
    return q.model_copy(update={"choices": choices, "correct_answer": str(choices.index(correct))})


def _answer_text(q: LLMQuestion) -> str:
    if q.question_type == QuestionType.MULTIPLE_CHOICE:
        idx = normalize_choice_answer(q.correct_answer, q.choices)
        return (q.choices or [""])[int(idx)] if idx is not None else ""
    return q.correct_answer


def _generate_round(req: GenerateRequest, summary: ParsedSummary, slots: list[Slot],
                    seen: list[str], strict: bool = True) -> list[LLMQuestion]:
    lines = summary.by_number()
    slot_by_index = {s.index: s for s in slots}

    def check(q: LLMQuestion, taken: list[str]) -> str | None:
        slot = slot_by_index.get(q.slot)
        if slot is None:
            return f"slot {q.slot}는 출제 계획에 없습니다."
        if q.question_type != slot.question_type:
            return f"slot {q.slot}: 유형은 {slot.question_type.value}여야 합니다."
        if not q.content.strip() or not q.explanation.strip():
            return f"slot {q.slot}: content 또는 explanation이 비어 있습니다."
        if any(same_text(q.content, s) for s in taken):
            return f"slot {q.slot}: 이미 낸 문제와 같습니다."
        ev = [lines[n] for n in q.evidence_lines if n in lines]
        if not ev or not quote_in_any(q.evidence_quote, [line.text for line in ev]):
            return f"slot {q.slot}: evidence_quote가 evidence_lines {q.evidence_lines}의 요약정리 원문에 없습니다."
        if slot.term is not None:
            key = normalize(slot.term.name)
            if not any(key in line.term_names for line in ev):
                return f"slot {q.slot}: 근거 줄이 용어 '{slot.term.name}'에 관한 줄이 아닙니다."
        if q.question_type == QuestionType.MULTIPLE_CHOICE:
            choices = [c.strip() for c in (q.choices or [])]
            if len(choices) != CHOICE_COUNT or not all(choices) or len({normalize(c) for c in choices}) != CHOICE_COUNT:
                return f"slot {q.slot}: 객관식은 서로 다른 보기 {CHOICE_COUNT}개가 필요합니다."
            if normalize_choice_answer(q.correct_answer, choices) is None:
                return f"slot {q.slot}: correct_answer는 보기 인덱스 \"0\"~\"{CHOICE_COUNT - 1}\"이어야 합니다."
        elif not q.correct_answer.strip():
            return f"slot {q.slot}: 주관식 모범 답안이 비어 있습니다."
        if strict and slot.term is not None and same_text(_answer_text(q).strip(" .?!요"), slot.term.name):
            # 정의를 주고 이름을 맞히는 문제는 이름만 외워도 풀린다 → 틀리게 배운 내용을 못 잡는다
            return (f"slot {q.slot}: 정답이 용어 이름 '{slot.term.name}' 자체입니다. 이름 맞히기 대신 "
                    f"요약정리에 적힌 특징·조건·결과·수치를 묻는 문제로 바꾸세요.")
        if strict and answer_leaked(_answer_text(q), q.content, ignore=slot.term.name if slot.term else ""):
            return f"slot {q.slot}: 문제 문장에 정답 내용이 거의 다 들어 있습니다. 정답을 드러내지 않게 다시 쓰세요."
        return None

    def validate(out: ExamGenerationOutput) -> Outcome[list[LLMQuestion]]:
        kept: dict[int, LLMQuestion] = {}
        fatal: list[str] = []
        taken = list(seen)
        for q in out.questions:
            if q.slot in kept:
                continue
            problem = check(q, taken)
            if problem:
                fatal.append(problem)
                continue
            if q.question_type == QuestionType.MULTIPLE_CHOICE:
                choices = [c.strip() for c in q.choices or []]
                q = q.model_copy(update={"choices": choices,
                                         "correct_answer": normalize_choice_answer(q.correct_answer, choices)})
            else:
                q = q.model_copy(update={"choices": None, "correct_answer": q.correct_answer.strip()})
            kept[q.slot] = q
            taken.append(q.content)
        missing = [s.index for s in slots if s.index not in kept]
        if missing:
            fatal.append(f"문제가 없는 slot: {missing}")
        return Outcome(list(kept.values()), fatal=fatal)

    plan = "\n".join(
        f"- slot {s.index}: 용어 {s.term.name if s.term else '(요약정리 전체)'}, 유형 {s.question_type.value}"
        for s in slots
    )
    audience = ("학생 AI가 수업에서 배운 내용으로 푸는 평가 문제다." if req.kind == ExamKind.EVALUATION
                else "사용자가 복습 차원에서 직접 푸는 재시험 문제다.")
    seen_block = "\n".join(f"- {s}" for s in seen) or "(없음)"
    system = render("exam_generate", audience=audience, rules=rules())
    user = f"[출제 계획]\n{plan}\n\n[요약정리 — 줄 번호]\n{summary.format()}\n\n[이미 낸 문제]\n{seen_block}"
    return call_structured(Task.EXAM_GENERATE, system=system, user=user,
                           output_model=ExamGenerationOutput, validate=validate).value


def _verify(questions: list[LLMQuestion], summary: ParsedSummary) -> dict[int, bool]:
    lines = summary.by_number()
    blocks = []
    for q in questions:
        choices = "".join(f"\n  [{i}] {c}" for i, c in enumerate(q.choices or []))
        evidence = "\n".join(f"  L{n}: {lines[n].text}" for n in q.evidence_lines if n in lines)
        blocks.append(f"[slot {q.slot}] ({q.question_type.value}) {q.content}{choices}\n"
                      f"  정답: {q.correct_answer}\n  근거:\n{evidence}")
    slots = {q.slot for q in questions}

    def validate(out: AnswerVerificationOutput) -> Outcome[dict[int, bool]]:
        result = {c.slot: c.answer_supported and c.single_correct_answer for c in out.checks if c.slot in slots}
        missing = sorted(slots - set(result))
        return Outcome(result, fatal=[f"검사가 빠진 slot: {missing}"] if missing else [])

    checked = call_structured(Task.EXAM_VERIFY, system=render("exam_verify"), user="\n\n".join(blocks),
                              output_model=AnswerVerificationOutput, validate=validate).value
    return {slot: checked.get(slot, False) for slot in slots}   # 검사가 빠지면 통과시키지 않음


def generate_exam(req: GenerateRequest) -> list[GeneratedQuestion]:
    summary = parse_summary(req.summary, [t.name for t in req.terms])
    if not summary.lines:
        raise OutputRejected("요약정리에 내용 줄이 없습니다.")
    slots = plan_slots(req, summary)
    seen = [q for q in req.exclude_questions if q.strip()]
    accepted: dict[int, LLMQuestion] = {}
    verify = get_settings().enable_answer_verification

    for round_no in range(1, MAX_ROUNDS + 1):
        todo = [s for s in slots if s.index not in accepted]
        if not todo:
            break
        # 마지막 라운드에서는 '이름 맞히기 금지'를 풀어서, 문제 수(count)를 못 채우는 일을 막는다
        made = _generate_round(req, summary, todo, seen + [q.content for q in accepted.values()],
                               strict=round_no < MAX_ROUNDS)
        passed = _verify(made, summary) if (verify and made) else {q.slot: True for q in made}
        for q in made:
            seen.append(q.content)   # 떨어진 문제도 다시 내지 않게
            if passed.get(q.slot):
                accepted[q.slot] = q

    if len(accepted) < len(slots):
        raise OutputRejected(f"검증을 통과한 문제가 {len(accepted)}/{len(slots)}개뿐입니다.")
    out = []
    for s in slots:
        q = _shuffle(accepted[s.index])
        out.append(GeneratedQuestion(
            number=s.index + 1, question_type=q.question_type, content=q.content.strip(),
            choices=q.choices, correct_answer=q.correct_answer, explanation=q.explanation.strip(),
            term_id=s.term.term_id if s.term else None,
        ))
    return out


# ───────────────────────── AI-7 학생 AI 문제 풀이 ─────────────────────────


def solve_exam(req: SolveRequest) -> list[SolveAnswer]:
    numbers = [q.number for q in req.questions]
    if len(numbers) != len(set(numbers)):
        raise validation_error("questions의 number가 중복되었어요")
    indexed = index_messages(req.messages)
    if not indexed.teacher:
        return [SolveAnswer(number=n, response=UNKNOWN_RESPONSE) for n in numbers]
    qmap = {q.number: q for q in req.questions}

    context = teacher_context(indexed)   # 답변에는 어느 질문에 대한 답인지 붙여서 검사 AI에 보여 준다
    verify_cache: dict[str, str | None] = {}

    def validate(out: ExamSolveOutput) -> Outcome[dict[int, str]]:
        fatal: list[str] = []
        answers: dict[int, str] = {}
        refs_by_number: dict[int, list[int]] = {}
        for a in out.answers:
            q = qmap.get(a.number)
            if q is None or a.number in answers:
                continue
            if not a.knows or not (a.response or "").strip():
                answers[a.number] = UNKNOWN_RESPONSE
                continue
            refs = [r for r in a.message_refs if r in indexed.teacher]
            if not refs or not quote_in_any(a.evidence_quote, [indexed.teacher[r] for r in refs]):
                fatal.append(f"{a.number}번: 답의 근거(선생님 메시지 원문 인용)가 확인되지 않습니다. 근거가 없으면 knows=false로 두세요.")
                answers[a.number] = UNKNOWN_RESPONSE
                continue
            if q.question_type == QuestionType.MULTIPLE_CHOICE:
                idx = normalize_choice_answer(a.response, q.choices)
                if idx is None:
                    fatal.append(f"{a.number}번: 객관식 답은 보기 인덱스 \"0\"~\"{len(q.choices or []) - 1}\"이어야 합니다.")
                    answers[a.number] = UNKNOWN_RESPONSE
                    continue
                answers[a.number] = idx
            else:
                bad_numbers = numbers_unsupported(a.response, list(indexed.teacher.values()))
                if bad_numbers:
                    fatal.append(f"{a.number}번: 답에 수업에서 듣지 않은 숫자({', '.join(bad_numbers)})가 있습니다. "
                                 "모르면 knows=false로 두세요.")
                    answers[a.number] = UNKNOWN_RESPONSE
                    continue
                answers[a.number] = a.response.strip()
            refs_by_number[a.number] = refs
        if claim_verification_on():
            known = [n for n in refs_by_number if answers.get(n) != UNKNOWN_RESPONSE]
            items = []
            for n in known:
                q = qmap[n]
                shown = answers[n]
                if q.question_type == QuestionType.MULTIPLE_CHOICE and q.choices:
                    shown = f"[{answers[n]}] {q.choices[int(answers[n])]}"
                items.append(f"문제: {q.content}\n학생 답: {shown}\n인용한 메시지: {', '.join(map(str, refs_by_number[n]))}")
            failed = verify_claims(Task.SOLVE_VERIFY, "solve_verify", context, items, verify_cache)
            for k, comment in sorted(failed.items()):
                n = known[k]
                fatal.append(f"{n}번: 검사 탈락 — {comment} 수업에서 들은 말로 답할 수 없으면 knows=false로 두세요.")
                answers[n] = UNKNOWN_RESPONSE
        for n in numbers:
            if n not in answers:
                fatal.append(f"{n}번: 답이 빠졌습니다.")
                answers[n] = UNKNOWN_RESPONSE
        return Outcome(answers, fatal=fatal)

    blocks = []
    for q in req.questions:
        choices = "".join(f"\n  [{i}] {c}" for i, c in enumerate(q.choices or []))
        blocks.append(f"{q.number}번 ({q.question_type.value}) {q.content}{choices}")
    system = student_system(req.persona, render("exam_solve"), voice="시험")
    user = f"[수업 대화]\n{indexed.format(req.persona.name)}\n\n[시험 문제]\n" + "\n".join(blocks)
    answers = call_structured(Task.EXAM_SOLVE, system=system, user=user, output_model=ExamSolveOutput,
                              validate=validate).value
    return [SolveAnswer(number=n, response=answers[n]) for n in numbers]
