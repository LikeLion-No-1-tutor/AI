"""여러 서비스가 함께 쓰는 프롬프트 조각과 수업 대화 정리."""
from __future__ import annotations

from collections.abc import Iterable
import logging
from dataclasses import dataclass

from ..config import get_settings
from ..errors import LLMError, LLMTimeout
from ..llm.router import Outcome, OutputRejected, Task, call_structured
from ..personas import voice_block
from ..logging_setup import log_event
from ..prompts import render
from ..rag.chunker import Chunk
from ..schemas.llm_outputs import ClaimVerificationOutput
from ..schemas.api import Message, Persona
from ..schemas.enums import MessageKind, Sender

UNKNOWN_RESPONSE = "모르겠어요. 수업에서 안 배웠어요."


def rules() -> str:
    return render("grounding_rules")


def student_system(persona: Persona, task_prompt: str, voice: str | None = None) -> str:
    """학생 AI 시스템 프롬프트 = Django의 persona.system_prompt + 학생 공통 규칙 + 이번 작업 (+ 글투).

    voice: 질문 / 노트 / 시험 — 이 페르소나의 글투(app/personas/voices.json)를 작업 뒤에 붙인다.
    손 들기 메모처럼 화면에 안 보이는 출력은 None (토큰 절약).
    """
    parts = [persona.system_prompt.strip(), render("student_rules"), task_prompt]
    block = voice_block(persona.code, persona.name, voice) if voice else ""
    if block:
        parts.append(block)
    return "\n\n".join(parts)


def format_chunks(chunks: Iterable[Chunk]) -> str:
    parts = [f'<chunk id="{c.id}" page="{c.page}">\n{c.text}\n</chunk>' for c in chunks]
    return "\n".join(parts) if parts else "(자료 청크 없음)"


@dataclass
class IndexedMessages:
    """수업 대화에 #1, #2 … 번호를 붙인 것. LLM은 이 번호로 근거를 가리킨다."""
    items: list[tuple[int, Message]]
    teacher: dict[int, str]          # 번호 → 선생님 메시지(설명·답변) 내용
    student_questions: list[str]

    def format(self, student_name: str = "학생") -> str:
        ref_by_id = {m.id: k for k, m in self.items if m.id is not None}
        lines = []
        for k, m in self.items:
            if m.sender == Sender.TEACHER and m.kind == MessageKind.ANSWER:
                target = ref_by_id.get(m.reply_to_id)
                label = f"선생님 답변 (#{target} 질문에 대해)" if target else "선생님 답변"
            elif m.sender == Sender.TEACHER:
                label = "선생님 설명"
            elif m.kind == MessageKind.QUESTION:
                label = f"{student_name} 질문"
            else:
                label = f"{student_name} 인사"
            lines.append(f"[#{k} {label}] {m.content}")
        return "\n".join(lines) if lines else "(대화 없음)"


def index_messages(messages: list[Message]) -> IndexedMessages:
    items = list(enumerate(messages, 1))
    teacher = {k: m.content for k, m in items if m.sender == Sender.TEACHER and m.content.strip()}
    questions = [m.content for _, m in items if m.kind == MessageKind.QUESTION]
    return IndexedMessages(items=items, teacher=teacher, student_questions=questions)


def teacher_context(indexed: IndexedMessages) -> str:
    """본문 검사(AI-5·AI-7)에 넘기는 선생님 메시지. 답변에는 어느 학생 질문에 대한 답인지 붙인다.

    질문 없이 답만 보여 주면 검사 AI가 "꼭 그렇진 않아" 같은 답을 앞의 다른 설명에 대한 반박으로
    잘못 읽고, 맞게 쓴 줄을 떨어뜨린다 (실측). 질문은 reply_to_id로, 없으면 바로 앞의 학생 질문으로 찾는다.
    """
    ref_by_id = {m.id: k for k, m in indexed.items if m.id is not None}
    by_ref = {k: m for k, m in indexed.items}
    lines, last_q = [], ""
    for k, m in indexed.items:
        if m.kind == MessageKind.QUESTION:
            last_q = m.content.strip()
        if k not in indexed.teacher:
            continue
        prefix = ""
        if m.kind == MessageKind.ANSWER:
            target = by_ref.get(ref_by_id.get(m.reply_to_id)) if m.reply_to_id is not None else None
            q = target.content.strip() if target is not None and target.kind == MessageKind.QUESTION else last_q
            prefix = f"(학생 질문 \"{q}\"에 대한 답) " if q else ""
        lines.append(f"[메시지 {k}] {prefix}{indexed.teacher[k]}")
    return "[선생님 메시지]\n" + "\n".join(lines)


# ───────────────────────── 본문 검사 (LLM) ─────────────────────────


def claim_verification_on() -> bool:
    return get_settings().enable_claim_verification


_NOT_CHECKED = "검사를 받지 못했습니다."
CHECK_ATTEMPTS = 2


def run_check(call, *, event: str, count: int, strict: bool = False, **fields):
    """검사 AI 호출. 호출 자체가 실패(LLM 오류·시간 초과·형식 불량)하면 한 번 더 부른다.

    두 번 다 실패하면: strict=False → None (호출한 쪽이 코드 검증만으로 통과시킴),
    strict=True → 마지막 오류를 그대로 올린다 (검사 없이 통과시키면 안 되는 용어 정의 등 → 502/504).
    어느 쪽이든 운영 로그에서 보이도록 WARNING으로 남긴다.
    """
    last: Exception | None = None
    for attempt in range(1, CHECK_ATTEMPTS + 1):
        try:
            return call()
        except (LLMError, LLMTimeout, OutputRejected) as exc:
            last = exc
            final = attempt == CHECK_ATTEMPTS
            result = ("요청 실패로 처리" if strict else f"새 항목 {count}개는 코드 검증만으로 통과") if final else "다시 부름"
            log_event(event, logging.WARNING, message=f"검사 호출 실패 (시도 {attempt}/{CHECK_ATTEMPTS}) → {result}",
                      error_code=getattr(exc, "code", "LLM_ERROR"), **fields)
    if strict:
        raise last
    return None


def verify_claims(task: Task, prompt: str, context: str, items: list[str],
                  cache: dict[str, str | None] | None = None, strict: bool = False) -> dict[int, str]:
    """만든 AI와 별개인 검사 호출 한 번으로, 항목마다 '주어진 근거만으로 이 문장이 나오는가'를 본다.

    인용 검증은 근거 문장이 원문에 있는지만 본다. 여기서는 본문이 그 근거와 같은 말인지를 본다.
    돌려주는 값: 떨어진 항목의 {번호: 이유}. 검사 응답에서 빠진 항목도 떨어뜨린다 (보수적).

    - cache: 바깥 validate가 다시 불릴 때 같은 문장을 또 검사하지 않도록, 호출하는 쪽이 validate 밖에서
      만든 dict를 넘긴다. {항목 본문: 탈락 이유 또는 None(통과)}.
    - 검사 호출 자체가 실패(LLM 오류·시간 초과)하면 한 번 더 부른다(run_check). 그래도 실패하면 이번에 새로 볼
      항목은 통과시키고 WARNING을 남긴다. 이 항목들은 이미 코드 검증(인용·숫자)을 통과했고, 검사 호출 하나 때문에
      다 만든 응답 전체를 502/504로 버리지 않기 위해서다 (요약·노트). 통과시킨 항목은 캐시에 넣지 않으므로,
      바깥이 다시 부르면 그때 다시 검사한다.
    - strict=True(AI-7 풀이, AI-9 보고서 검사): 두 번 다 실패하면 통과시키지 않고 오류를 올린다.
    """
    if not items:
        return {}
    cache = {} if cache is None else cache
    todo = [i for i, body in enumerate(items) if body not in cache]
    missing: dict[int, str] = {}
    if todo:
        failed = run_check(lambda: _check(task, prompt, context, [items[i] for i in todo]),
                           event="claim_verification_skipped", count=len(todo), strict=strict, task=task.value)
        if failed is None:
            failed = {}
        else:
            for k, i in enumerate(todo):
                if failed.get(k) == _NOT_CHECKED:
                    missing[i] = _NOT_CHECKED   # 이번엔 떨어뜨리되, 다음 시도에 다시 검사받게 캐시하지 않는다
                else:
                    cache[items[i]] = failed.get(k)
    out = {i: cache[body] for i, body in enumerate(items) if cache.get(body)}
    out.update(missing)
    return out


def _check(task: Task, prompt: str, context: str, items: list[str]) -> dict[int, str]:
    wanted = set(range(len(items)))

    def validate(out: ClaimVerificationOutput) -> Outcome[dict[int, str]]:
        failed: dict[int, str] = {}
        seen: set[int] = set()
        for c in out.checks:
            if c.index not in wanted or c.index in seen:
                continue
            seen.add(c.index)
            if not c.supported:
                failed[c.index] = c.comment.strip() or "근거와 맞지 않습니다."
        missing = sorted(wanted - seen)
        for i in missing:
            failed[i] = _NOT_CHECKED
        return Outcome(failed, fatal=[f"검사가 빠진 index: {missing}"] if missing else [])

    user = f"{context}\n\n[검사할 항목]\n" + "\n\n".join(f"[index {i}]\n{body}" for i, body in enumerate(items))
    return call_structured(task, system=render(prompt), user=user,
                           output_model=ClaimVerificationOutput, validate=validate).value
