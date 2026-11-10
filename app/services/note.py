"""AI-5 정리노트 작성 (역할: student, 일반).

노트는 세 부분이다 (팀 결정).
  1. 개념 설명      — 선생님 '설명' 메시지를 주제별로
  2. 질문과 답      — 학생 질문(원문, 코드가 붙임) + 선생님 '답변'을 정리한 줄
  3. 아직 궁금한 사항 — 손 들었지만 선생님이 받지 않은 이유(raised_reasons) + 답을 못 받은 학생 질문

- 학생 AI가 페르소나 말투로, 선생님이 말한 내용만 쓴다 (틀린 내용도 그대로).
- 줄마다 '선생님 메시지 원문 인용'을 받아 코드로 확인한다 → 학생이 자기 지식을 몰래 적는 것을 막는다.
- 본문 검사: 다른 호출이 줄 내용이 선생님 말만으로 나오는지 다시 본다 (ENABLE_CLAIM_VERIFICATION).
  틀리게 가르친 내용은 '선생님이 그렇게 말했으면' 통과한다.
- 질문과 답의 짝, 세 부분의 순서, Markdown은 코드가 정한다.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from ..grounding import normalize, numbers_unsupported, quote_in_any, split_sentences, unheard_terms
from ..llm.router import Outcome, Task, call_structured, is_last_attempt
from ..logging_setup import log_event
from ..personas import note_check
from ..prompts import render
from ..schemas.api import NoteRequest
from ..schemas.enums import MessageKind, Sender
from ..schemas.llm_outputs import NoteLine, NoteOutput, NoteSection
from .common import (IndexedMessages, claim_verification_on, index_messages, rules, student_system,
                     teacher_context, verify_claims)

EMPTY_NOTE_LINE = "아직 선생님께 배운 내용이 없어요."
EMPTY_QA_LINE = "수업 중에 주고받은 질문이 없어요."
EMPTY_WONDER_LINE = "남은 궁금한 점이 없어요."
HEAD_CONCEPT, HEAD_QA, HEAD_WONDER = "개념 설명", "질문과 답", "아직 궁금한 사항"
FALLBACK_HEADING = "배운 내용"

# 누락 검사: 선생님 문장마다, 그 메시지를 근거로 쓴 노트 줄들(본문+인용)에 글자 쌍(bigram)이 이만큼 있어야 담긴 것으로 본다
COVER_RATIO = 0.5
MIN_SENTENCE_CHARS = 4      # 정규화 후 이보다 짧은 문장('응', '좋아')은 보지 않음
# 수업 예고 문장("오늘은 경사하강법을 배울 거야.")은 내용이 아니라서 빠뜨려도 다시 부르지 않는다
# (실측: 거의 매번 이 문장 때문에 다시 불렸고, 억지로 넣은 줄이 어색해졌다)
_INTRO = re.compile(r"^(자,?\s*)?(오늘|이번\s*시간|지금부터|이제부터)(은|는|에는|에)?\s.*(배울|배워\s*볼|알아볼|공부할|살펴볼|해\s*볼)")


def _bigrams(text: str) -> set[str]:
    t = normalize(text)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def missing_sentences(teacher: dict[int, str], lines: list[NoteLine]) -> list[str]:
    """노트에 담기지 않은 선생님 문장(설명·답변). 물음(?로 끝남)과 아주 짧은 말은 보지 않는다.

    학생 AI가 의심스럽다고 스스로 거르는 것(예: '보통 다 잘 맞혀.')을 막는다. 틀린 말도 노트에 남아야 한다.
    """
    pool: dict[int, set[str]] = {}
    for ln in lines:
        for r in ln.message_refs:
            pool.setdefault(r, set()).update(_bigrams(ln.text) | _bigrams(ln.evidence_quote))
    out = []
    for k, content in teacher.items():
        for s in split_sentences(content):
            if len(normalize(s)) < MIN_SENTENCE_CHARS or s.endswith(("?", "？")) or _INTRO.search(s.strip()):
                continue
            g = _bigrams(s)
            if g and len(g & pool.get(k, set())) / len(g) < COVER_RATIO:
                out.append(s)
    return out


# 소제목 검사: 3글자 이상 단어가 수업 대화에 한 번도 안 나왔으면 학생이 사전 지식으로 붙인 용어로 본다.
# (예: 선생님이 '경사하강법'이라고 말한 적 없는데 소제목에 '경사하강법'을 씀)
_WORD = re.compile(r"[가-힣A-Za-z0-9]+")
_JOSA = ("이란", "에서", "으로", "과", "와", "은", "는", "이", "가", "을", "를", "의", "란", "로", "도", "및")
_GENERIC = {"알아보기", "정리하기", "마무리", "업데이트", "기초", "개념", "방법", "원리", "조건", "이유", "예시",
            "특징", "차이", "관계", "정리", "종료", "무엇", "과정", "핵심", "주의", "비교", "결과", "선생님", "질문", "답변",
            "비유", "비유하면", "비유하기", "요약", "한줄", "전체", "오늘", "수업", "주제"}


def _heading_words_unheard(heading: str, heard: str) -> list[str]:
    unheard = []
    for word in _WORD.findall(heading):
        stem = word
        for josa in _JOSA:
            if stem.endswith(josa) and len(stem) - len(josa) >= 2:
                stem = stem[: -len(josa)]
                break
        if len(stem) < 3 or stem in _GENERIC or stem.startswith(("비유", "정리", "요약", "설명")):
            continue
        # 활용형은 인정한다: '내려간다'를 들었으면 '내려가는'·'내려가기'도 들은 말
        # (끝 1~2글자를 떼고 3글자 이상 남긴 앞부분이 들은 말에 있으면. 남은 끝 글자는 받침만 달라도 됨: 가 ↔ 간)
        norm = normalize(stem)
        if not any(_heard_prefix(norm[:k], heard) for k in range(len(norm), max(3, len(norm) - 2) - 1, -1)):
            unheard.append(word)
    return unheard


def _heard_prefix(prefix: str, heard: str) -> bool:
    if prefix in heard:
        return True
    last = prefix[-1]
    if not "가" <= last <= "힣":
        return False
    base = (ord(last) - 0xAC00) // 28 * 28 + 0xAC00   # 받침 없는 같은 글자
    return re.search(re.escape(prefix[:-1]) + f"[{chr(base)}-{chr(base + 27)}]", heard) is not None


def _voice_problem(persona_code: str, lines: list[NoteLine]) -> str | None:
    """페르소나 글투 표시(voices.json의 note_check)가 있는 줄이 너무 적으면 다시 쓰라는 메시지.

    말투가 평범한 '~라고 하셨다'로 수렴하는 것을 막는다. 마지막 시도에서는 보지 않는다 (말투 때문에 노트가 실패하지 않게).
    """
    rule = note_check(persona_code)
    if not rule or not lines:
        return None
    pattern = re.compile(rule["regex"])
    ratio = sum(1 for ln in lines if pattern.search(ln.text.strip())) / len(lines)
    if ratio >= rule["min_ratio"]:
        return None
    return f"글투가 페르소나답지 않습니다 ({ratio:.0%}만 해당). {rule['hint']}"


# ───────────────────────── 질문과 답 짝 짓기 (코드) ─────────────────────────


@dataclass
class QAPair:
    answer_ref: int          # 선생님 답변 메시지 번호
    question: str            # 학생 질문 원문 (학생이 실제로 한 말이라 그대로). 빈 문자열 = 앞 질문에 이어진 답변
    lines: list[NoteLine] = field(default_factory=list)


def pair_questions(indexed: IndexedMessages) -> tuple[list[QAPair], list[str], set[int]]:
    """답변 메시지마다 어느 질문에 대한 답인지 정한다.

    reply_to_id가 있으면 그 질문, 없으면 바로 앞의 아직 짝이 없는 질문.
    돌려주는 값: (짝, 답을 못 받은 질문들, 질문을 못 찾은 답변 번호 — 개념 설명으로 돌린다).
    """
    ref_by_id = {m.id: k for k, m in indexed.items if m.id is not None}
    questions = {k: m.content for k, m in indexed.items if m.kind == MessageKind.QUESTION and m.content.strip()}
    used: set[int] = set()
    pairs: list[QAPair] = []
    orphan: set[int] = set()
    for k, m in indexed.items:
        if m.sender != Sender.TEACHER or m.kind != MessageKind.ANSWER or k not in indexed.teacher:
            continue
        q = ref_by_id.get(m.reply_to_id) if m.reply_to_id is not None else None
        if q not in questions:
            earlier = [i for i in questions if i < k and i not in used]
            q = earlier[-1] if earlier else None
        if q is None:
            orphan.add(k)
            continue
        if q in used:   # 같은 질문에 답변을 여러 메시지로 나눠 보냄 → 앞 질문 아래에 이어 붙인다
            pairs.append(QAPair(answer_ref=k, question=""))
            continue
        used.add(q)
        pairs.append(QAPair(answer_ref=k, question=questions[q]))
    unanswered = [questions[i] for i in questions if i not in used]
    return pairs, unanswered, orphan


# ───────────────────────── Markdown 조립 (코드) ─────────────────────────


def render_note(title: str, sections: list[NoteSection], qa: list[QAPair] | None = None,
                wonders: list[str] | None = None) -> str:
    out = [f"# {title} 정리노트", "", f"## {HEAD_CONCEPT}", ""]
    if not sections:
        out += [EMPTY_NOTE_LINE, ""]
    for i, section in enumerate(sections, 1):
        out.append(f"### {i}. {section.heading.strip()}")
        out += [f"- {line.text.strip()}" for line in section.lines]
        out.append("")

    out += [f"## {HEAD_QA}", ""]
    shown = [p for p in (qa or []) if p.lines]
    if not shown:
        out += [EMPTY_QA_LINE, ""]
    n = 0
    for p in shown:
        if p.question:
            if n:
                out.append("")
            n += 1
            out.append(f"### Q{n}. {p.question.strip()}")
        out += [f"- {line.text.strip()}" for line in p.lines]
    if shown:
        out.append("")

    out += [f"## {HEAD_WONDER}", ""]
    out += [f"- {w.strip()}" for w in (wonders or [])] or [EMPTY_WONDER_LINE]
    return "\n".join(out).rstrip() + "\n"


# ───────────────────────── AI-5 ─────────────────────────


def write_note(req: NoteRequest) -> str:
    indexed = index_messages(req.messages)
    pairs, unanswered, _orphan = pair_questions(indexed)
    reasons = [r.strip() for r in req.raised_reasons if r.strip()]
    if not indexed.teacher:
        return render_note(req.title, [], [], reasons + unanswered)

    answer_refs = {p.answer_ref for p in pairs}
    explain = {k: v for k, v in indexed.teacher.items() if k not in answer_refs}   # 질문을 못 찾은 답변도 여기로
    heard = normalize(" ".join(m.content for _, m in indexed.items))
    heard_all = [m.content for _, m in indexed.items] + [req.persona.system_prompt]
    taught = list(indexed.teacher.values())   # 노트 줄 내용은 선생님이 한 말 안에서만
    context = teacher_context(indexed)   # 답변에는 어느 질문에 대한 답인지 붙여서 검사 AI에 보여 준다
    verify_cache: dict[str, str | None] = {}
    dropped = 0

    def check_line(where: str, line: NoteLine, allowed: dict[int, str], fatal: list[str]) -> NoteLine | None:
        if not line.text.strip():
            return None
        refs = [r for r in line.message_refs if r in allowed]
        if not refs:
            fatal.append(f"{where}: message_refs {line.message_refs}는 이 부분에 쓸 수 있는 선생님 메시지가 아닙니다 "
                         f"(쓸 수 있는 번호: {sorted(allowed)}).")
            return None
        if not quote_in_any(line.evidence_quote, [allowed[r] for r in refs]):
            fatal.append(f"{where}: evidence_quote가 선생님 메시지 원문에 없습니다.")
            return None
        bad_numbers = numbers_unsupported(line.text, taught)
        if bad_numbers:
            fatal.append(f"{where}: 선생님이 말하지 않은 숫자({', '.join(bad_numbers)})가 있습니다.")
            return None
        return line.model_copy(update={"message_refs": refs})

    def validate(out: NoteOutput) -> Outcome[tuple[list[NoteSection], list[QAPair], list[str]]]:
        nonlocal dropped
        fatal: list[str] = []
        total = 0

        # ① 개념 설명 (설명 메시지만)
        sections: list[NoteSection] = []
        for s_idx, section in enumerate(out.concept_sections, 1):
            kept = []
            for l_idx, line in enumerate(section.lines, 1):
                total += 1
                ok = check_line(f"개념 {s_idx}-{l_idx}번 줄", line, explain, fatal)
                if ok:
                    kept.append(ok)
            heading = section.heading.strip()
            unheard = _heading_words_unheard(heading, heard)
            if kept and unheard:
                fatal.append(f"개념 {s_idx}번 소제목 '{heading}': 수업 대화에 나오지 않은 말({', '.join(unheard)})이 "
                             "있습니다. 선생님이 쓴 말로만 소제목을 쓰세요.")
                heading = FALLBACK_HEADING   # 다시 불러도 고쳐지지 않으면 이 제목으로
            if kept:
                sections.append(section.model_copy(update={"heading": heading or FALLBACK_HEADING, "lines": kept}))

        # ② 질문과 답 (답변 메시지마다, 그 답변만 근거로)
        by_ref = {p.answer_ref for p in pairs}
        got: dict[int, list[NoteLine]] = {}
        for a in out.answers:
            if a.message_ref not in by_ref:
                fatal.append(f"answers의 message_ref {a.message_ref}는 [질문과 답 목록]의 답변 번호가 아닙니다.")
                continue
            allowed = {a.message_ref: indexed.teacher[a.message_ref]}
            for l_idx, line in enumerate(a.lines, 1):
                total += 1
                ok = check_line(f"답변 #{a.message_ref}의 {l_idx}번 줄", line, allowed, fatal)
                if ok:
                    got.setdefault(a.message_ref, []).append(ok)
        qa = [QAPair(p.answer_ref, p.question, list(got.get(p.answer_ref, []))) for p in pairs]

        # 본문 검사 (개념 줄 + 답변 줄을 한 번에)
        if claim_verification_on():
            flat = [ln for sec in sections for ln in sec.lines] + [ln for p in qa for ln in p.lines]
            items = [f"노트 줄: {ln.text}\n인용한 메시지: {', '.join(map(str, ln.message_refs))}" for ln in flat]
            failed = verify_claims(Task.NOTE_VERIFY, "note_verify", context, items, verify_cache)
            for k, comment in sorted(failed.items()):
                fatal.append(f"줄 '{flat[k].text[:40]}': 검사 탈락 — {comment} 선생님이 말한 내용만 적으세요.")
            drop = {id(flat[k]) for k in failed}
            sections = [sec.model_copy(update={"lines": [ln for ln in sec.lines if id(ln) not in drop]})
                        for sec in sections]
            sections = [sec for sec in sections if sec.lines]
            for p in qa:
                p.lines = [ln for ln in p.lines if id(ln) not in drop]
        empty = [p.answer_ref for p in qa if not p.lines]
        if empty:
            fatal.append(f"질문과 답: 답변 #{', #'.join(map(str, empty))}의 정리가 없습니다. 답변마다 정리하세요.")

        # ③ 아직 궁금한 사항 (메모를 페르소나 말투로. 끝내 못 쓰면 메모 원문)
        wonder_text: dict[int, str] = {}
        for w in out.wonders:
            text = w.text.strip()
            if not (1 <= w.index <= len(reasons)) or not text or w.index in wonder_text:
                continue
            latin = [t for t in unheard_terms(text, heard_all + [reasons[w.index - 1]]) if t.isascii()]
            if latin:
                fatal.append(f"아직 궁금한 점 {w.index}번: 수업에서 나오지 않은 말({', '.join(latin)})이 있습니다.")
                continue
            if text.count("?") + text.count("？") > 1:
                fatal.append(f"아직 궁금한 점 {w.index}번: 궁금한 점 하나만 한 문장으로 쓰세요.")
                continue
            wonder_text[w.index] = text
        missing_w = [i for i in range(1, len(reasons) + 1) if i not in wonder_text]
        if missing_w:
            fatal.append(f"아직 궁금한 점 {missing_w}번이 없습니다. 메모마다 하나씩 쓰세요.")
        wonders = [wonder_text.get(i, reasons[i - 1]) for i in range(1, len(reasons) + 1)] + unanswered

        lines_all = [ln for sec in sections for ln in sec.lines] + [ln for p in qa for ln in p.lines]
        if not lines_all:
            fatal.append("노트가 비어 있습니다. 선생님이 말한 내용을 정리하세요.")
        elif not is_last_attempt():
            missing = missing_sentences(indexed.teacher, lines_all)
            if missing:
                shown = " / ".join(f"\"{m[:60]}\"" for m in missing[:5])
                fatal.append(f"선생님이 말한 문장 {len(missing)}개가 노트에 없습니다: {shown}. "
                             "틀리거나 의심스러운 말이라도 빠뜨리지 말고 선생님 말 그대로 적으세요.")
            voice_problem = _voice_problem(req.persona.code, lines_all)
            if voice_problem:
                fatal.append(voice_problem)
        dropped = total - len(lines_all)
        return Outcome((sections, qa, wonders), fatal=fatal)

    explain_list = ", ".join(f"#{k}" for k in explain) or "(없음)"
    qa_list = "\n".join(
        f"- 답변 #{p.answer_ref} ← 질문: {p.question or '(바로 앞 답변과 같은 질문에 이어서 한 답)'}" for p in pairs
    ) or "(없음)"
    wonder_list = "\n".join(f"{i}. {r}" for i, r in enumerate(reasons, 1)) or "(없음)"
    system = student_system(req.persona, render("note", rules=rules()), voice="노트")
    user = (f"[수업 주제] {req.title}\n\n[수업 대화]\n{indexed.format(req.persona.name)}\n\n"
            f"[① 개념 설명에 쓸 선생님 설명 메시지] {explain_list}\n\n"
            f"[② 질문과 답 목록]\n{qa_list}\n\n"
            f"[③ 아직 궁금한 점 — 손 들었지만 선생님이 받지 않은 메모]\n{wonder_list}")
    sections, qa, wonders = call_structured(Task.NOTE, system=system, user=user, output_model=NoteOutput,
                                            validate=validate).value
    # 정리 줄이 하나도 남지 않은 답변은 선생님 답을 그대로 둔다 (선생님이 한 말은 노트에 남아야 함)
    for p in qa:
        if not p.lines:
            p.lines = [NoteLine(text=f"선생님 답: {indexed.teacher[p.answer_ref].strip()}",
                                message_refs=[p.answer_ref], evidence_quote="")]
    if dropped:
        log_event("note_lines_dropped", logging.DEBUG, dropped=dropped)
    return render_note(req.title, sections, qa, wonders)
