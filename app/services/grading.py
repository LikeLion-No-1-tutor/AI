"""AI-8 주관식 채점 (evaluator, 심화). 맞음·틀림만, 부분 점수 없음.

- 객관식은 Django가 직접 채점하므로 주관식만 들어온다.
- 답이 비었거나 '모르겠어요'인 문항은 LLM을 부르지 않고 코드가 틀림 처리.
- 문항이 하나라도 판정되지 않으면 추측하지 않고 502 LLM_ERROR (점수에 직접 들어가므로).
"""
from __future__ import annotations

import logging

from ..errors import validation_error
from ..grounding import same_text
from ..llm.router import Outcome, Task, call_structured
from ..logging_setup import log_event
from ..prompts import render
from ..schemas.api import GradeRequest, GradeResult
from ..schemas.llm_outputs import GradingOutput
from .common import UNKNOWN_RESPONSE

NO_ANSWER_REASON = "답을 내지 못했어요."


def grade(req: GradeRequest) -> list[GradeResult]:
    numbers = [i.number for i in req.items]
    if len(numbers) != len(set(numbers)):
        raise validation_error("items의 number가 중복되었어요")

    results: dict[int, GradeResult] = {}
    to_llm = []
    for item in req.items:
        response = (item.response or "").strip()
        if not response or same_text(response, UNKNOWN_RESPONSE):
            results[item.number] = GradeResult(number=item.number, is_correct=False, reason=NO_ANSWER_REASON)
        else:
            to_llm.append(item)

    if to_llm:
        wanted = {i.number for i in to_llm}

        def validate(out: GradingOutput) -> Outcome[dict[int, GradeResult]]:
            fatal: list[str] = []
            got: dict[int, GradeResult] = {}
            for g in out.results:
                if g.number not in wanted:
                    continue
                if g.number in got:
                    fatal.append(f"number {g.number}을 두 번 채점했습니다. 문항마다 한 번만 채점하세요.")
                    continue
                got[g.number] = GradeResult(number=g.number, is_correct=g.is_correct, reason=g.reason.strip())
            missing = sorted(wanted - set(got))
            if missing:
                fatal.append(f"채점이 빠진 number: {missing}")
            return Outcome(got, fatal=fatal)

        blocks = [f"{i.number}번 문제: {i.content}\n  모범 답안: {i.correct_answer}\n  해설: {i.explanation}\n"
                  f"  제출한 답: {i.response}" for i in to_llm]
        graded = call_structured(Task.EXAM_GRADE, system=render("exam_grade"), user="\n\n".join(blocks),
                                 output_model=GradingOutput, validate=validate, accept_after_retries=False).value
        results.update(graded)
        log_event("grade_reasons", logging.DEBUG, results=[r.model_dump() for r in graded.values()])

    return [results[n] for n in numbers]
