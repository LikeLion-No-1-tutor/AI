"""AI-5 정리노트 작성 (역할: student, 일반).

- 학생 AI가 페르소나 말투로, 선생님이 말한 내용만 쓴다 (틀린 내용도 그대로).
- 줄마다 '선생님 메시지 원문 인용'을 받아 코드로 확인한다 → 학생이 자기 지식을 몰래 적는 것을 막는다.
- 본문 검사: 다른 호출이 줄 내용이 선생님 말만으로 나오는지 다시 본다 (ENABLE_CLAIM_VERIFICATION).
  틀리게 가르친 내용은 '선생님이 그렇게 말했으면' 통과한다.
- Markdown은 코드가 조립한다.
"""
from __future__ import annotations

import logging
import re

from ..grounding import normalize, numbers_unsupported, quote_in_any, split_sentences
from ..llm.router import Outcome, Task, call_structured, is_last_attempt
from ..logging_setup import log_event
from ..prompts import render
from ..schemas.api import NoteRequest
from ..schemas.llm_outputs import NoteOutput, NoteSection
from .common import claim_verification_on, index_messages, rules, student_system, verify_claims

EMPTY_NOTE_LINE = "아직 선생님께 배운 내용이 없어요."

# 누락 검사: 선생님 설명 문장마다, 그 메시지를 근거로 쓴 노트 줄들(본문+인용)에 글자 쌍(bigram)이 이만큼 있어야 담긴 것으로 본다
COVER_RATIO = 0.5
MIN_SENTENCE_CHARS = 4      # 정규화 후 이보다 짧은 문장('응', '좋아')은 보지 않음


def _bigrams(text: str) -> set[str]:
    t = normalize(text)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def missing_sentences(teacher: dict[int, str], sections: list[NoteSection]) -> list[str]:
    """노트에 담기지 않은 선생님 설명 문장. 물음(?로 끝남)과 아주 짧은 말은 보지 않는다.

    학생 AI가 의심스럽다고 스스로 거르는 것(예: '보통 다 잘 맞혀.')을 막는다. 틀린 말도 노트에 남아야 한다.
    """
    pool: dict[int, set[str]] = {}
    for sec in sections:
        for ln in sec.lines:
            for r in ln.message_refs:
                pool.setdefault(r, set()).update(_bigrams(ln.text) | _bigrams(ln.evidence_quote))
    out = []
    for k, content in teacher.items():
        for s in split_sentences(content):
            if len(normalize(s)) < MIN_SENTENCE_CHARS or s.endswith(("?", "？")):
                continue
            g = _bigrams(s)
            if g and len(g & pool.get(k, set())) / len(g) < COVER_RATIO:
                out.append(s)
    return out
FALLBACK_HEADING = "배운 내용"

# 소제목 검사: 3글자 이상 단어가 수업 대화에 한 번도 안 나왔으면 학생이 사전 지식으로 붙인 용어로 본다.
# (예: 선생님이 '경사하강법'이라고 말한 적 없는데 소제목에 '경사하강법'을 씀)
_WORD = re.compile(r"[가-힣A-Za-z0-9]+")
_JOSA = ("이란", "에서", "으로", "과", "와", "은", "는", "이", "가", "을", "를", "의", "란", "로", "도", "및")
_GENERIC = {"알아보기", "정리하기", "마무리", "업데이트", "기초", "개념", "방법", "원리", "조건", "이유", "예시",
            "특징", "차이", "관계", "정리", "종료", "무엇", "과정", "핵심", "주의", "비교", "결과", "선생님", "질문", "답변"}


def _heading_words_unheard(heading: str, heard: str) -> list[str]:
    unheard = []
    for word in _WORD.findall(heading):
        stem = word
        for josa in _JOSA:
            if stem.endswith(josa) and len(stem) - len(josa) >= 2:
                stem = stem[: -len(josa)]
                break
        if len(stem) >= 3 and stem not in _GENERIC and normalize(stem) not in heard:
            unheard.append(word)
    return unheard


def render_note(title: str, sections: list[NoteSection]) -> str:
    out = [f"# {title} 정리노트", ""]
    if not sections:
        out.append(EMPTY_NOTE_LINE)
    for i, section in enumerate(sections, 1):
        out += [f"## {i}. {section.heading.strip()}"]
        out += [f"- {line.text.strip()}" for line in section.lines]
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def write_note(req: NoteRequest) -> str:
    indexed = index_messages(req.messages)
    if not indexed.teacher:
        return render_note(req.title, [])

    dropped = 0
    heard = normalize(" ".join(m.content for _, m in indexed.items))
    taught = list(indexed.teacher.values())   # 노트 줄 내용은 선생님이 한 말 안에서만
    context = "[선생님 메시지]\n" + "\n".join(f"[메시지 {k}] {v}" for k, v in indexed.teacher.items())
    verify_cache: dict[str, str | None] = {}

    def validate(out: NoteOutput) -> Outcome[list[NoteSection]]:
        nonlocal dropped
        fatal: list[str] = []
        sections: list[NoteSection] = []
        total = kept_count = 0
        for s_idx, section in enumerate(out.sections, 1):
            kept = []
            for l_idx, line in enumerate(section.lines, 1):
                total += 1
                if not line.text.strip():
                    continue
                refs = [r for r in line.message_refs if r in indexed.teacher]
                if not refs:
                    fatal.append(f"{s_idx}-{l_idx}번 줄: message_refs {line.message_refs}가 선생님 메시지가 아닙니다.")
                    continue
                if not quote_in_any(line.evidence_quote, [indexed.teacher[r] for r in refs]):
                    fatal.append(f"{s_idx}-{l_idx}번 줄: evidence_quote가 선생님 메시지 원문에 없습니다.")
                    continue
                bad_numbers = numbers_unsupported(line.text, taught)
                if bad_numbers:
                    fatal.append(f"{s_idx}-{l_idx}번 줄: 선생님이 말하지 않은 숫자({', '.join(bad_numbers)})가 있습니다.")
                    continue
                kept.append(line)
            heading = section.heading.strip()
            unheard = _heading_words_unheard(heading, heard)
            if kept and unheard:
                fatal.append(f"{s_idx}번 소제목 '{heading}': 수업 대화에 나오지 않은 말({', '.join(unheard)})이 있습니다. "
                             "선생님이 쓴 말로만 소제목을 쓰세요.")
                heading = FALLBACK_HEADING   # 다시 불러도 고쳐지지 않으면 이 제목으로
            if kept:
                sections.append(section.model_copy(update={"heading": heading or FALLBACK_HEADING, "lines": kept}))
                kept_count += len(kept)
        if sections and claim_verification_on():
            flat = [ln for sec in sections for ln in sec.lines]
            items = [f"노트 줄: {ln.text}\n인용한 메시지: {', '.join(map(str, ln.message_refs))}" for ln in flat]
            failed = verify_claims(Task.NOTE_VERIFY, "note_verify", context, items, verify_cache)
            for k, comment in sorted(failed.items()):
                fatal.append(f"줄 '{flat[k].text[:40]}': 검사 탈락 — {comment} 선생님이 말한 내용만 적으세요.")
            drop = {id(flat[k]) for k in failed}
            sections = [sec.model_copy(update={"lines": [ln for ln in sec.lines if id(ln) not in drop]}) for sec in sections]
            sections = [sec for sec in sections if sec.lines]
            kept_count -= len(failed)
        if not sections:
            fatal.append("노트가 비어 있습니다. 선생님이 말한 내용을 정리하세요.")
        elif not is_last_attempt():
            missing = missing_sentences(indexed.teacher, sections)
            if missing:
                shown = " / ".join(f"\"{m[:60]}\"" for m in missing[:5])
                fatal.append(f"선생님이 설명한 문장 {len(missing)}개가 노트에 없습니다: {shown}. "
                             "틀리거나 의심스러운 말이라도 빠뜨리지 말고 선생님 말 그대로 적으세요.")
        dropped = total - kept_count
        return Outcome(sections, fatal=fatal)

    system = student_system(req.persona, render("note", rules=rules()))
    user = f"[수업 주제] {req.title}\n\n[수업 대화]\n{indexed.format(req.persona.name)}"
    sections = call_structured(Task.NOTE, system=system, user=user, output_model=NoteOutput, validate=validate).value
    if dropped:
        log_event("note_lines_dropped", logging.DEBUG, dropped=dropped)
    return render_note(req.title, sections)
