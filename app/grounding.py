"""근거 검증 (할루시네이션 방지의 핵심).

LLM이 '이 문장이 근거다'라고 인용(evidence_quote)하면, 그 인용이 실제로
지정한 청크·대화 턴 안에 있는지 코드로 확인한다. 없으면 그 항목은 지어낸 것으로 보고 버린다.

비교 전에 공백·문장부호·대소문자를 지우고(정규화), 정확히 포함되거나
(긴 인용만) 90% 이상 같은 연속 구간이 있어야 통과한다. 비슷하게만 맞을 때 부정어가 다르면 떨어뜨린다.

인용이 진짜여도 본문이 인용과 다른 말을 할 수 있다. 본문 내용은 services의 LLM 검사(verify_claims)가 보고,
여기서는 LLM이 약한 숫자만 numbers_unsupported로 코드 대조한다.
"""
from __future__ import annotations

import re
from functools import lru_cache
import unicodedata
from difflib import SequenceMatcher

_STRIP = re.compile(r"[\s\W_]+", re.UNICODE)

MIN_QUOTE_CHARS = 8       # 정규화 후 이보다 짧은 인용은 근거로 인정하지 않음 (원문 자체가 짧으면 원문 전체). 12는 짧은 식·라벨 인용을 막아 실측에서 오탐
QUOTE_FLOOR_CHARS = 6     # 원문이 아무리 짧아도 이보다 짧은 인용은 인정하지 않음
FUZZY_MIN_CHARS = 20      # 이보다 짧은 인용은 글자 그대로 포함될 때만 인정
# 비슷하게만 맞는 인용에서 이 말의 개수가 원문과 다르면 뜻이 뒤집힌 것으로 본다 (정규화된 글자 기준)
_NEGATIONS = ("않", "못", "없", "아니", "안", "not", "never", "cannot")


def normalize(text: str) -> str:
    return _STRIP.sub("", unicodedata.normalize("NFKC", text or "").lower())


def _negation_differs(a: str, b: str) -> bool:
    return any(a.count(n) != b.count(n) for n in _NEGATIONS)


SENTENCE_FLOOR_CHARS = 4  # 원문의 '한 문장 전체'를 그대로 인용했으면 이 길이까지 인정 ("보통 다 잘 맞혀.")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。？！])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    """문장 단위로 나눈다 (마침표·느낌표·물음표 뒤 공백, 줄바꿈)."""
    return [s.strip() for s in _SENTENCE_SPLIT.split(text or "") if s.strip()]


def quote_in_text(quote: str | None, text: str, min_ratio: float = 0.9) -> bool:
    q, t = normalize(quote or ""), normalize(text)
    if not q or not t:
        return False
    if len(q) < max(QUOTE_FLOOR_CHARS, min(MIN_QUOTE_CHARS, len(t))):
        # 짧은 인용은 원문의 한 문장 전체와 똑같을 때만 인정한다 (문장 조각 '학습률' 같은 건 여전히 안 됨)
        return len(q) >= SENTENCE_FLOOR_CHARS and any(q == normalize(s) for s in split_sentences(text))
    if q in t:
        return True
    if len(q) < FUZZY_MIN_CHARS:
        return False
    match = SequenceMatcher(None, q, t, autojunk=False).find_longest_match(0, len(q), 0, len(t))
    if match.size / len(q) < min_ratio:
        return False
    start = max(0, match.b - match.a)
    return not _negation_differs(q, t[start:start + len(q)])


def quote_in_any(quote: str | None, texts: list[str]) -> bool:
    return any(quote_in_text(quote, t) for t in texts)


def same_text(a: str, b: str) -> bool:
    return normalize(a) == normalize(b)


COVERED_RATIO = 0.72   # 이 비율 이상이 다른 문장에 이미 들어 있으면 '같은 말'로 본다
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def covered_by(text: str, other: str, ignore: str = "") -> bool:
    """text의 내용이 other에 이미 들어 있는지 (요약정리 중복 제거용).

    - ignore(보통 용어 이름)는 양쪽에서 지우고 비교한다. "기울기는 …"처럼 주어만 붙은 문장을 잡기 위해.
    - 3글자 이상 이어서 같은 부분의 합이 text의 COVERED_RATIO 이상이면 같은 말.
    - 숫자가 3개 이상이고 숫자 묶음이 똑같으면 같은 예시로 본다 (예: 3, 2, 0.1, 2.8).
    """
    key = normalize(ignore)
    a, b = normalize(text), normalize(other)
    if key:
        a, b = a.replace(key, ""), b.replace(key, "")
    if not a:
        return True
    if not b:
        return False
    nums_a, nums_b = set(_NUMBER.findall(text)), set(_NUMBER.findall(other))
    if len(nums_a) >= 3 and nums_a == nums_b:
        return True
    return _overlap(a, b) >= COVERED_RATIO


def _overlap(a: str, b: str) -> float:
    blocks = SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks()
    return sum(m.size for m in blocks if m.size >= 3) / len(a)


ANSWER_LEAK_RATIO = 0.6


def answer_leaked(answer: str, question: str, ignore: str = "") -> bool:
    """문제 문장이 정답 내용을 대부분 담고 있으면 True (정답을 알려주는 문제). 용어 이름은 빼고 비교."""
    key = normalize(ignore)
    a, q = normalize(answer), normalize(question)
    if key:
        a, q = a.replace(key, ""), q.replace(key, "")
    return len(a) >= 8 and bool(q) and _overlap(a, q) >= ANSWER_LEAK_RATIO


_PAREN = re.compile(r"[(\[（](.*?)[)\]）]")


def term_aliases(name: str) -> list[str]:
    """용어 이름의 여러 표기 (정규화). 예: '순환 신경망(RNN)' → ['순환신경망rnn', '순환신경망', 'rnn'].

    괄호 안팎을 따로 인정한다. 본 이름(전체·괄호 밖)은 한 글자도 인정하고('큐', '힙'),
    괄호 안 보조 표기는 2글자 이상만 쓴다 (예: '스택(S)'의 'S'로는 판단하지 않음).
    """
    outside = _PAREN.sub(" ", name)
    inside = [part for group in _PAREN.findall(name) for part in re.split(r"[,/·]", group)]
    out: list[str] = []
    for alias, min_len in [(name, 1), (outside, 1), *[(part, 2) for part in inside]]:
        key = normalize(alias)
        if len(key) >= min_len and key not in out:
            out.append(key)
    return out


@lru_cache(maxsize=512)
def _latin_pattern(alias: str) -> re.Pattern[str]:
    # 'timernn'은 'Time RNN', 'Time-RNN'에도 맞게 글자 사이 공백·기호를 허용. 앞뒤는 영숫자가 아니어야 한다
    # (그래야 'ai'가 'email'에, 'rnn'이 'birnn'에 맞지 않는다). 복수형 s는 허용.
    body = r"[\s\-_./]*".join(re.escape(ch) for ch in alias)
    return re.compile(rf"(?<![a-z0-9]){body}s?(?![a-z0-9])")


def mentions_term(name: str, text: str) -> bool:
    """글에 용어(또는 그 다른 표기)가 나오는가. 영문·숫자로만 된 표기는 단어 경계로, 한글이 섞인 표기는 글자 포함으로 본다."""
    body = normalize(text)
    lowered = None
    for alias in term_aliases(name):
        if alias.isascii() and alias.isalnum():
            if lowered is None:
                lowered = unicodedata.normalize("NFKC", text or "").lower()
            if _latin_pattern(alias).search(lowered):
                return True
        elif len(alias) == 1:
            # 한 글자 한글 용어('큐', '힙'): '큐브'·'힙합' 속 글자가 아니라 낱말일 때만 (뒤에 조사는 허용)
            if lowered is None:
                lowered = unicodedata.normalize("NFKC", text or "").lower()
            if _single_pattern(alias).search(lowered):
                return True
        elif alias in body:
            return True
    return False


_ONE_CHAR_JOSA = ("으로|에서|까지|부터|처럼|보다|이랑|라는|라고|이라|은|는|이|가|을|를|의|에|와|과|도|로|만|랑|란|나|야")


@lru_cache(maxsize=128)
def _single_pattern(alias: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![가-힣]){re.escape(alias)}(?=$|[^가-힣]|(?:{_ONE_CHAR_JOSA})(?:$|[^가-힣]))")


# ───────────────────────── 숫자 대조 · 질문 용어 검사 ─────────────────────────

_HANGUL_WORD = re.compile(r"[가-힣]+")
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*")
_JOSAS = tuple(sorted({
    "에서는", "으로는", "이라는", "이라고", "에게서", "까지는", "부터는",
    "이란", "에서", "으로", "에게", "까지", "부터", "보다", "처럼", "라는", "라고", "에는", "와는", "과는", "이나",
    "은요", "는요", "이랑", "하고",
    "과", "와", "은", "는", "이", "가", "을", "를", "의", "란", "로", "도", "및", "만", "에", "나", "랑",
}, key=len, reverse=True))
_COPULAS = tuple(sorted({"이에요", "예요", "이야", "인가요", "인데요", "인가", "인데", "이죠", "이에", "이다", "입니다"},
                        key=len, reverse=True))
# 근거와 상관없이 쓸 수 있는 말 (설명·서술용, 수업 진행용). 낱말이 이것으로 시작하면 검사하지 않는다.
_FILLER = (
    "의미", "나타내", "나타낸", "사용", "때문", "이용", "통해", "통한", "위해", "위한", "경우", "따라", "따른", "대한", "대해",
    "관련", "가능", "필요", "중요", "역할", "방식", "방법", "기능", "목적", "문제", "해결", "결과", "과정", "특징", "개념",
    "정의", "설명", "말씀", "선생님", "수업", "배웠", "배운", "배우", "알게", "알려", "들었", "이해", "궁금", "모르", "헷갈",
    "그러면", "그런데", "그래서", "그렇다면", "그렇게", "그러니까", "그럼", "어떻게", "어떤", "무엇", "무슨", "왜냐", "정말", "혹시",
    "다시", "조금", "아까", "똑같", "다른", "차이", "관계", "예시", "예를", "이유", "원리", "조건", "주의", "비교", "핵심",
    "있는", "있다", "없는", "없다", "하는", "한다", "된다", "되는", "이다", "이러", "이런", "이렇", "저런", "그런", "그것", "이것",
    "여기", "거기", "우리", "모든", "각각", "서로", "함께", "또한", "그리고", "하지만", "그러나", "즉", "만약", "만일",
)
# 서술어 끝 (AI-4 질문 검사에서 동사·형용사로 보고 건너뛴다)
_VERB_ENDINGS = ("요", "까", "죠", "니", "나", "다", "데", "면", "서", "고", "게", "지", "해", "네", "걸", "던", "며", "도록",
                 "어야", "아야", "해야")


def _strip_josa(word: str) -> str:
    for suffix in _COPULAS + _JOSAS:
        if word.endswith(suffix) and len(word) - len(suffix) >= 2:
            return word[: -len(suffix)]
    return word


def _is_filler(stem: str) -> bool:
    return stem.startswith(_FILLER)


def _found(stem: str, source: str, min_prefix: int) -> bool:
    """낱말 그대로, 또는 끝 몇 글자(어미)를 뗀 앞부분이 원문에 있으면 True."""
    return any(stem[:k] in source for k in range(len(stem), min_prefix - 1, -1))


def _numbers(text: str) -> set[str]:
    """두 자리 이상이거나 소수인 숫자만 (한 자리 정수는 '2가지'처럼 셈에 흔해서 뺀다)."""
    return {n for n in _NUMBER.findall(text or "") if len(n) >= 2 or "." in n}


def numbers_unsupported(text: str, sources: list[str]) -> list[str]:
    """본문의 숫자(두 자리 이상·소수) 중 근거 원문에 없는 것. 한 자리 정수는 '2가지'처럼 셈에 흔해서 보지 않는다.

    숫자가 원문 어딘가에 '있는지'만 본다. 원문의 다른 숫자와 바꿔 붙인 것은 LLM 검사(verify_claims)가 본다.
    """
    src = _numbers(" ".join(sources))
    return [n for n in sorted(_numbers(text)) if n not in src]


def unheard_terms(text: str, heard: list[str]) -> list[str]:
    """학생 질문(AI-4)에 들은 적 없는 전문 용어가 있으면 그 말 목록.

    heard = 수업 대화 + 페르소나 프롬프트('원래 아는 것'). 영문 용어는 엄격하게,
    한글은 서술어(…요, …까 등)와 설명용 말을 빼고 남은 3글자 이상 낱말을 본다.
    """
    src = normalize(" ".join(heard))
    out = []
    for word in dict.fromkeys(w.lower() for w in _LATIN_WORD.findall(text or "")):
        if len(word) >= 2 and word not in src:
            out.append(word)
    for word in dict.fromkeys(_HANGUL_WORD.findall(text or "")):
        stem = _strip_josa(word)
        if len(stem) < 3 or _is_filler(stem) or stem.endswith(_VERB_ENDINGS):
            continue
        if not _found(stem, src, max(2, len(stem) - 1)):
            out.append(word)
    return out
