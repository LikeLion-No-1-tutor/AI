"""AI-5 정리노트 작성 (역할: student, 일반).

노트는 체크리스트 용어마다 나뉜다 (프론트 정리노트 화면 기준, 2026-11-11).
  용어 ─ 개념               — 선생님 '설명' 메시지를 학생 AI가 페르소나 말투로 정리한 줄
       ─ 학생 AI의 질문      — 학생 질문(원문) + 나의 답변(선생님 답 원문). 둘 다 코드가 그대로 붙인다
       ─ 확인하지 않은 질문   — 손 들었지만 선생님이 받지 않은 메모(raised_reasons)를 질문으로 + 답을 못 받은 학생 질문
  어느 용어에도 맞지 않는 내용은 마지막 '그 밖에 배운 내용'에 모은다.

- 학생 AI가 페르소나 말투로, 선생님이 말한 내용만 쓴다 (틀린 내용도 그대로).
- 개념 줄마다 '선생님 메시지 원문 인용'을 받아 코드로 확인한다 → 학생이 자기 지식을 몰래 적는 것을 막는다.
- 본문 검사: 다른 호출이 줄 내용이 선생님 말만으로 나오는지 다시 본다 (ENABLE_CLAIM_VERIFICATION).
  틀리게 가르친 내용은 '선생님이 그렇게 말했으면' 통과한다.
- LLM은 질문이 어느 용어에 대한 것인지만 정한다. 질문과 답의 짝, 순서, Markdown은 코드가 정한다.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from ..grounding import _negation_differs, normalize, numbers_unsupported, quote_in_any, split_sentences, unheard_terms
from ..llm.router import Outcome, Task, call_structured, is_last_attempt
from ..logging_setup import log_event
from ..personas import note_check
from ..prompts import render
from ..schemas.api import NoteQuestion, NoteRequest, NoteTermSection
from ..schemas.enums import MessageKind, Sender
from ..schemas.llm_outputs import NoteLine, NoteOutput
from .common import (IndexedMessages, claim_verification_on, index_messages, rules, student_system,
                     teacher_context, verify_claims)

OTHER = 0                                   # LLM이 '어느 용어에도 맞지 않음'에 쓰는 term_id
OTHER_NAME = "그 밖에 배운 내용"
HEAD_CONCEPT, HEAD_ASKED, HEAD_UNSEEN = "개념", "학생 AI의 질문", "확인하지 않은 질문"
# 다른 용어 아래에 함께 정리됐을 수도 있어서 '안 배웠다'고 단정하지 않는다
# (실측: 체크리스트는 '후입선출' explained인데 내용이 '스택' 아래에 들어가 "아직 배우지 않았어요"가 나옴)
EMPTY_CONCEPT_LINE = "이 용어로 따로 정리한 내용은 없어요."
ASKED_ONLY_LINE = "아래 질문과 답에서 배웠어요."
EMPTY_NOTE_LINE = "아직 선생님께 배운 내용이 없어요."

# 누락 검사: 선생님 문장마다, 그 메시지를 근거로 쓴 노트 줄들(본문+인용)에 글자 쌍(bigram)이 이만큼 있어야 담긴 것으로 본다
COVER_RATIO = 0.5
MIN_SENTENCE_CHARS = 4      # 정규화 후 이보다 짧은 문장('응', '좋아')은 보지 않음
# 수업 예고 문장("오늘은 경사하강법을 배울 거야.")은 내용이 아니라서 빠뜨려도 다시 부르지 않는다
# (실측: 거의 매번 이 문장 때문에 다시 불렸고, 억지로 넣은 줄이 어색해졌다)
_INTRO = re.compile(r"^(자,?\s*)?(오늘|이번\s*시간|지금부터|이제부터)(은|는|에는|에)?\s.*(배울|배워\s*볼|알아볼|공부할|살펴볼|해\s*볼)")
# 순서만 알리는 짧은 말("오늘은 스택부터 할게.", "다음은 큐야.")도 내용이 아니다
# (실측: 빠짐 검사 때문에 "다음 순서는 큐예요" 같은 줄이 개념에 들어갔다)
_TRANSITION = re.compile(r"^(자,?\s*)?(오늘은|이번에는|이번엔|다음은|다음으로|그럼|이제)\s*\S{1,15}?"
                         r"(부터\s*(할게|하자|볼게|시작할게|시작하자)|(이)?야|(으)?로\s*넘어가(자|볼게|겠습니다)?)[.!]?$")
# 학생에게 차례를 넘기는 말("궁금한 거 있으면 물어봐.")도 내용이 아니다. 물음표가 없어서 빠짐 검사에 걸렸다
# (실측: 3회 중 2회 "궁금한 거 있으면 물어보래요."가 개념 줄에 들어가고, 매번 첫 시도가 떨어짐).
# lesson.py의 _HANDOFF_END는 '끝나', '여기까지'처럼 설명 문장에도 나오는 말을 잡으므로 쓰지 않고,
# 문장 전체가 넘김 말일 때만 맞도록 좁게 잡는다
_HANDOFF_SENTENCE = re.compile(
    r"^(자,?\s*|그럼,?\s*|혹시,?\s*)*"
    r"(((궁금한|모르는|헷갈리는|이해\s*안\s*되는|이해가\s*안\s*되는)\s*(거|것|게|점|부분)(이|가)?|질문(이)?)\s*(있으면|있다면)"
    r"|이해\s*안\s*되면|이해가\s*안\s*되면|모르겠으면|헷갈리면)"
    r"\s*(언제든지?\s*)?(다시\s*)?"
    r"(물어봐|물어보세요|물어봐도\s*돼|질문해|질문하세요|말해|말해줘|말씀하세요|얘기해|해|해줘)(요)?\s*[.!~]*$")


def _bigrams(text: str) -> set[str]:
    t = normalize(text)
    return {t[i:i + 2] for i in range(len(t) - 1)}


# 중복 줄: 짧은 줄의 글자 쌍이 긴 줄에 이만큼 있고, 긴 줄에 없는 글자 쌍이 이 개수 이하일 때만 같은 줄로 본다.
# 일부러 엄격하게 잡았다 — 한 낱말만 다른 줄은 뜻이 다르다 (실측: "학습률이 높으면/낮으면 발산함"은 빠진 쌍 2개,
# "분산을/과적합을 줄인대요"는 3개라서 남고, "이렇게 나누는 걸 분할이라고 해/한대요!"는 1개라서 지운다).
DUP_RATIO = 0.85
DUP_MAX_EXTRA = 1
# 확인하지 않은 질문은 같은 궁금증으로 손을 여러 번 들어 생긴다. 질문 하나를 빼도 내용이 사라지지 않으므로 느슨하게 본다
# (같은 용어 안에서만. 실측: "나무 결과를 어떻게 합치는지 궁금"과 "나무들 결과는 어떻게 합치는지 궁금해요"는 0.71)
WONDER_DUP_RATIO = 0.6


def _overlap(a: str, b: str) -> tuple[float, int]:
    """(짧은 쪽 글자 쌍 중 긴 쪽에도 있는 비율, 짧은 쪽에만 있는 글자 쌍 수)."""
    x, y = _bigrams(a), _bigrams(b)
    short, long_ = (x, y) if len(x) <= len(y) else (y, x)
    if not short:
        return 0.0, 0
    return len(short & long_) / len(short), len(short - long_)


def is_duplicate_line(a: str, b: str) -> bool:
    ratio, extra = _overlap(a, b)
    return ratio >= DUP_RATIO and extra <= DUP_MAX_EXTRA and not _negation_differs(normalize(a), normalize(b))


def is_duplicate_wonder(a: str, b: str) -> bool:
    return _overlap(a, b)[0] >= WONDER_DUP_RATIO


def dedupe(items: list[tuple[int, str]], same, same_term_only: bool) -> list[tuple[int, str]]:
    """(용어, 글) 목록에서 비슷한 것끼리는 긴 글 하나만 남긴다. 자리(용어·순서)는 먼저 나온 쪽."""
    keep: list[tuple[int, str]] = []
    for term, text in items:
        for i, (k_term, kept) in enumerate(keep):
            if (not same_term_only or k_term == term) and same(text, kept):
                if len(normalize(text)) > len(normalize(kept)):
                    keep[i] = (k_term, text)
                break
        else:
            keep.append((term, text))
    return keep


def missing_sentences(teacher: dict[int, str], lines: list[NoteLine]) -> list[str]:
    """노트에 담기지 않은 선생님 문장. 물음(?로 끝남), 아주 짧은 말, 수업 예고·순서 안내·넘김 말은 보지 않는다.

    학생 AI가 의심스럽다고 스스로 거르는 것(예: '보통 다 잘 맞혀.')을 막는다. 틀린 말도 노트에 남아야 한다.
    """
    pool: dict[int, set[str]] = {}
    for ln in lines:
        for r in ln.message_refs:
            pool.setdefault(r, set()).update(_bigrams(ln.text) | _bigrams(ln.evidence_quote))
    out = []
    said: list[str] = []
    for k, content in teacher.items():
        for s in split_sentences(content):
            if (len(normalize(s)) < MIN_SENTENCE_CHARS or s.endswith(("?", "？")) or _INTRO.search(s.strip())
                    or _TRANSITION.search(s.strip()) or _HANDOFF_SENTENCE.search(s.strip())):
                continue
            # 앞에서 한 말을 되풀이한 문장("다시 말하지만 이렇게 나누는 걸 분할이라고 해.")은 다시 적으라고 하지 않는다
            # (실측: 이 검사 때문에 같은 줄이 노트에 세 번 들어갔다). 기준은 중복 줄과 같이 엄격하게 —
            # "지니 불순도가 0이면/1이면", "과적합이 줄어/심해져"처럼 바꿔 말한 틀린 설명은 건너뛰면 안 된다
            repeated = any(is_duplicate_line(s, p) for p in said)
            said.append(s)
            if repeated:
                continue
            g = _bigrams(s)
            if g and len(g & pool.get(k, set())) / len(g) < COVER_RATIO:
                out.append(s)
    return out


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
    question: str                                        # 학생 질문 원문 (학생이 실제로 한 말이라 그대로)
    answer_refs: list[int] = field(default_factory=list)  # 선생님 답변 메시지 번호. 비어 있으면 답을 못 받은 질문


def pair_questions(indexed: IndexedMessages) -> tuple[list[QAPair], set[int]]:
    """학생 질문마다 선생님 답변을 붙인다 (수업 순서).

    답변은 reply_to_id가 있으면 그 질문, 없으면 바로 앞의 아직 답을 못 받은 질문.
    같은 질문에 답변을 여러 메시지로 나눠 보냈으면 모두 그 질문에 붙인다.
    돌려주는 값: (질문 목록, 질문을 못 찾은 답변 번호 — 개념으로 돌린다).
    """
    ref_by_id = {m.id: k for k, m in indexed.items if m.id is not None}
    pairs: dict[int, QAPair] = {k: QAPair(m.content) for k, m in indexed.items
                                if m.kind == MessageKind.QUESTION and m.content.strip()}
    orphan: set[int] = set()
    for k, m in indexed.items:
        if m.sender != Sender.TEACHER or m.kind != MessageKind.ANSWER or k not in indexed.teacher:
            continue
        q = ref_by_id.get(m.reply_to_id) if m.reply_to_id is not None else None
        if q not in pairs:
            waiting = [i for i in pairs if i < k and not pairs[i].answer_refs]
            q = waiting[-1] if waiting else None
        if q is None:
            orphan.add(k)
            continue
        pairs[q].answer_refs.append(k)
    return list(pairs.values()), orphan


# ───────────────────────── 화면용 구조 + Markdown (코드) ─────────────────────────


@dataclass
class _Section:
    term_id: int | None
    name: str
    concept: list[str] = field(default_factory=list)
    questions: list[NoteQuestion] = field(default_factory=list)


def build_sections(req: NoteRequest, concept_lines: dict[int, list[str]], asked: list[tuple[int, NoteQuestion]],
                   unseen: list[tuple[int, NoteQuestion]]) -> list[NoteTermSection]:
    """용어 순서대로 섹션을 만든다. 질문은 답변 완료 → 확인하지 않은 질문 순."""
    sections: dict[int, _Section] = {t.term_id: _Section(t.term_id, t.name) for t in req.terms}
    other = _Section(None, OTHER_NAME if req.terms else req.title)

    def target(term_id: int) -> _Section:
        return sections.get(term_id, other)

    for term_id, lines in concept_lines.items():
        target(term_id).concept.extend(lines)
    for term_id, q in asked + unseen:
        target(term_id).questions.append(q)
    out = list(sections.values())
    if other.concept or other.questions:
        out.append(other)
    return [NoteTermSection(term_id=s.term_id, name=s.name, concept="\n".join(s.concept), questions=s.questions)
            for s in out]


def _one_line(text: str) -> str:
    return " ".join(text.split())


def render_note(title: str, sections: list[NoteTermSection]) -> str:
    out = [f"# {title} 정리노트", ""]
    if not any(s.concept or s.questions for s in sections):
        return "\n".join(out + [EMPTY_NOTE_LINE]) + "\n"
    for i, s in enumerate(sections, 1):
        out += [f"## {i:02d}. {s.name}", "", f"### {HEAD_CONCEPT}", ""]
        asked = [q for q in s.questions if q.status == "answered"]
        unseen = [q for q in s.questions if q.status == "unseen"]
        empty = ASKED_ONLY_LINE if asked else EMPTY_CONCEPT_LINE   # 질문에 답하며 가르친 용어는 '안 배움'이 아니다
        out += [f"- {line}" for line in s.concept.split("\n") if line.strip()] or [empty]
        out.append("")
        if asked:
            out += [f"### {HEAD_ASKED}", ""]
            for n, q in enumerate(asked, 1):
                out += [f"- **Q{n}.** {_one_line(q.question)}", f"  - 나의 답변: {_one_line(q.answer or '')}"]
            out.append("")
        if unseen:
            out += [f"### {HEAD_UNSEEN}", ""]
            out += [f"- {_one_line(q.question)}" for q in unseen]
            out.append("")
    return "\n".join(out).rstrip() + "\n"


# ───────────────────────── AI-5 ─────────────────────────


def write_note(req: NoteRequest) -> tuple[str, list[NoteTermSection]]:
    """돌려주는 값: (Markdown, 화면용 용어별 구조)."""
    indexed = index_messages(req.messages)
    pairs, _orphan = pair_questions(indexed)
    reasons = [r.strip() for r in req.raised_reasons if r.strip()]
    term_ids = {t.term_id for t in req.terms}
    valid_ids = term_ids | {OTHER}

    answer_refs = {r for p in pairs for r in p.answer_refs}
    explain = {k: v for k, v in indexed.teacher.items() if k not in answer_refs}   # 질문을 못 찾은 답변도 여기로

    def question_items(assign: dict[int, int]) -> tuple[list[tuple[int, NoteQuestion]], list[tuple[int, NoteQuestion]]]:
        asked, unseen = [], []
        for n, p in enumerate(pairs, 1):
            term = assign.get(n, OTHER)
            if p.answer_refs:
                answer = "\n".join(indexed.teacher[r].strip() for r in p.answer_refs)
                asked.append((term, NoteQuestion(status="answered", question=p.question.strip(), answer=answer)))
            else:
                unseen.append((term, NoteQuestion(status="unseen", question=p.question.strip(), answer=None)))
        return asked, unseen

    if not indexed.teacher and not reasons and not pairs:
        sections = build_sections(req, {}, [], [])
        return render_note(req.title, sections), sections

    heard_all = [m.content for _, m in indexed.items] + [req.persona.system_prompt]
    taught = list(indexed.teacher.values())   # 노트 줄 내용은 선생님이 한 말 안에서만
    context = teacher_context(indexed)   # 답변에는 어느 질문에 대한 답인지 붙여서 검사 AI에 보여 준다
    verify_cache: dict[str, str | None] = {}
    dropped = 0

    def check_line(where: str, line: NoteLine, fatal: list[str]) -> NoteLine | None:
        if not line.text.strip():
            return None
        refs = [r for r in line.message_refs if r in explain]
        if not refs:
            fatal.append(f"{where}: message_refs {line.message_refs}는 개념에 쓸 수 있는 선생님 설명 메시지가 아닙니다 "
                         f"(쓸 수 있는 번호: {sorted(explain)}).")
            return None
        if not quote_in_any(line.evidence_quote, [explain[r] for r in refs]):
            fatal.append(f"{where}: evidence_quote가 선생님 메시지 원문에 없습니다.")
            return None
        bad_numbers = numbers_unsupported(line.text, taught)
        if bad_numbers:
            fatal.append(f"{where}: 선생님이 말하지 않은 숫자({', '.join(bad_numbers)})가 있습니다.")
            return None
        return line.model_copy(update={"message_refs": refs})

    def term_of(where: str, term_id: int, fatal: list[str]) -> int:
        if term_id in valid_ids:
            return term_id
        fatal.append(f"{where}: term_id {term_id}는 [용어 목록]에 없습니다 (쓸 수 있는 번호: {sorted(valid_ids)}).")
        return OTHER

    def validate(out: NoteOutput) -> Outcome[tuple[dict[int, list[str]], dict[int, int], list[tuple[int, NoteQuestion]]]]:
        nonlocal dropped
        fatal: list[str] = []
        total = 0

        # ① 개념 (설명 메시지만, 용어별)
        kept: list[tuple[int, NoteLine]] = []
        for c_idx, concept in enumerate(out.concepts, 1):
            term = term_of(f"개념 {c_idx}번 항목", concept.term_id, fatal)
            for l_idx, line in enumerate(concept.lines, 1):
                total += 1
                ok = check_line(f"개념 {c_idx}-{l_idx}번 줄", line, fatal)
                if ok:
                    kept.append((term, ok))

        # 본문 검사
        if claim_verification_on() and kept:
            items = [f"노트 줄: {ln.text}\n인용한 메시지: {', '.join(map(str, ln.message_refs))}" for _, ln in kept]
            failed = verify_claims(Task.NOTE_VERIFY, "note_verify", context, items, verify_cache)
            for k, comment in sorted(failed.items()):
                fatal.append(f"줄 '{kept[k][1].text[:40]}': 검사 탈락 — {comment} 선생님이 말한 내용만 적으세요.")
            kept = [x for k, x in enumerate(kept) if k not in failed]

        # ② 학생 AI의 질문 → 용어 (끝내 못 정하면 '그 밖에')
        assign: dict[int, int] = {}
        for q in out.questions:
            if 1 <= q.number <= len(pairs) and q.number not in assign:
                assign[q.number] = term_of(f"질문 {q.number}번", q.term_id, fatal)
        missing_q = [n for n in range(1, len(pairs) + 1) if n not in assign]
        if missing_q:
            fatal.append(f"[질문 목록] {missing_q}번의 term_id가 없습니다. 질문마다 하나씩 쓰세요.")

        # ③ 확인하지 않은 질문 (메모를 페르소나 말투의 질문으로. 끝내 못 쓰면 메모 원문)
        wonder: dict[int, tuple[int, str]] = {}
        wonder_term: dict[int, int] = {}   # 문장이 떨어져도 용어는 살린다 (메모 원문을 그 용어 아래에)
        for w in out.wonders:
            text = w.text.strip()
            if not (1 <= w.index <= len(reasons)) or not text or w.index in wonder:
                continue
            term = term_of(f"메모 {w.index}번", w.term_id, fatal)
            wonder_term.setdefault(w.index, term)
            latin = [t for t in unheard_terms(text, heard_all + [reasons[w.index - 1]]) if t.isascii()]
            if latin:
                fatal.append(f"메모 {w.index}번: 수업에서 나오지 않은 말({', '.join(latin)})이 있습니다.")
                continue
            marks = text.count("?") + text.count("？")
            if marks != 1:
                fatal.append(f"메모 {w.index}번: 선생님께 묻는 질문 하나를 물음표 한 개로 끝나는 한 문장으로 쓰세요.")
                continue
            wonder[w.index] = (term, text)
        missing_w = [i for i in range(1, len(reasons) + 1) if i not in wonder]
        if missing_w:
            fatal.append(f"메모 {missing_w}번이 없습니다. 메모마다 하나씩 쓰세요.")
        wonders = []
        for i in range(1, len(reasons) + 1):
            term, text = wonder.get(i, (wonder_term.get(i, OTHER), reasons[i - 1]))
            wonders.append((term, NoteQuestion(status="unseen", question=text, answer=None)))

        lines_all = [ln for _, ln in kept]
        if explain and not lines_all:
            fatal.append("개념이 비어 있습니다. 선생님이 설명한 내용을 정리하세요.")
        elif lines_all and not is_last_attempt():
            missing = missing_sentences(explain, lines_all)
            if missing:
                shown = " / ".join(f"\"{m[:60]}\"" for m in missing[:5])
                fatal.append(f"선생님이 말한 문장 {len(missing)}개가 노트에 없습니다: {shown}. "
                             "틀리거나 의심스러운 말이라도 빠뜨리지 말고 선생님 말 그대로 적으세요.")
            voice_problem = _voice_problem(req.persona.code, lines_all)
            if voice_problem:
                fatal.append(voice_problem)
            # 개념 칸은 배운 내용을 적는 곳이라 물음이 있으면 '확인하지 않은 질문'과 헷갈린다
            # (실측: 민준 질문 글투 "제 말로 하면 ~거죠?"가 개념 줄에 들어감). 마지막 시도에서는 그대로 둔다
            asking = [ln.text for ln in lines_all if "?" in ln.text or "？" in ln.text]
            if asking:
                shown = " / ".join(f"\"{t[:40]}\"" for t in asking[:3])
                fatal.append(f"개념 줄에 물음표가 있습니다: {shown}. 개념 줄은 배운 내용을 적는 문장으로 쓰고 "
                             "물음표를 쓰지 마세요 (\"~거죠?\" → \"~인 거다\").")
        dropped = total - len(lines_all)

        concept_lines: dict[int, list[str]] = {}
        # 감탄 뒤 쉼표("오!, 큐는…")는 글투 예시 때문에 자주 붙는 군더더기라 정리한다
        cleaned = [(term, re.sub(r"([!！])\s*,", r"\1", ln.text.strip())) for term, ln in kept]
        # 거의 그대로 반복된 줄은 하나만 (실측: '분할' 설명이 세 번, 회귀나무·랜덤포레스트 줄이 반복됨).
        # 빠짐 검사(missing_sentences)는 위에서 지우기 전 줄로 했으므로 영향 없음
        for term, text in dedupe(cleaned, is_duplicate_line, same_term_only=False):
            concept_lines.setdefault(term, []).append(text)
        # 같은 궁금증으로 손을 여러 번 들면 거의 같은 질문이 여러 개 생긴다 → 같은 용어 안에서 하나로
        merged = dedupe([(t, q.question) for t, q in wonders], is_duplicate_wonder, same_term_only=True)
        wonders = [(t, NoteQuestion(status="unseen", question=text, answer=None)) for t, text in merged]
        return Outcome((concept_lines, assign, wonders), fatal=fatal)

    term_list = "\n".join(f"- term_id {t.term_id}: {t.name}" for t in req.terms) or "(용어 없음 — 모두 0)"
    explain_list = ", ".join(f"#{k}" for k in explain) or "(없음)"
    q_list = "\n".join(f"{n}. {p.question.strip()}" for n, p in enumerate(pairs, 1)) or "(없음)"
    wonder_list = "\n".join(f"{i}. {r}" for i, r in enumerate(reasons, 1)) or "(없음)"
    system = student_system(req.persona, render("note", rules=rules()), voice="노트")
    user = (f"[수업 주제] {req.title}\n\n[용어 목록]\n{term_list}\n- term_id 0: 어느 용어에도 맞지 않음\n\n"
            f"[수업 대화]\n{indexed.format(req.persona.name)}\n\n"
            f"[① 개념에 쓸 선생님 설명 메시지] {explain_list}\n\n"
            f"[② 질문 목록]\n{q_list}\n\n"
            f"[③ 손 들었지만 묻지 못한 메모]\n{wonder_list}")
    concept_lines, assign, wonders = call_structured(Task.NOTE, system=system, user=user, output_model=NoteOutput,
                                                     validate=validate).value
    asked, unseen = question_items(assign)
    if dropped:
        log_event("note_lines_dropped", logging.DEBUG, dropped=dropped)
    sections = build_sections(req, concept_lines, asked, unseen + wonders)
    return render_note(req.title, sections), sections
