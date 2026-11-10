"""요약정리 Markdown 만들기와 읽기.

만들기(AI-2): 코드가 조립한다 → AI가 형식이나 내용을 바꿀 여지가 없다.
읽기(AI-6): Django가 저장해 둔 Markdown을 줄 단위로 번호(L1, L2 …)를 붙이고,
            각 줄이 어느 용어에 속하는지 찾는다. 출제 AI는 이 줄 번호로 근거를 가리킨다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..grounding import normalize, pair_parts


@dataclass
class SummaryTerm:
    name: str
    definition: str
    points: list[tuple[str, list[int]]] = field(default_factory=list)   # (문장, 쪽)


def _pages(pages: list[int]) -> str:
    return f" (p.{', '.join(map(str, pages))})" if pages else ""


def build_summary_markdown(title: str, terms: list[SummaryTerm],
                           relations: list[tuple[str, list[int]]]) -> str:
    out = [f"# {title} 요약정리", "", "## 핵심 개념", ""]
    for t in terms:
        out += [f"### {t.name}", f"- 정의: {t.definition}"]
        out += [f"- {text}{_pages(pages)}" for text, pages in t.points]
        out.append("")
    if relations:
        out += ["## 개념 간 관계", ""]
        out += [f"- {text}{_pages(pages)}" for text, pages in relations]
        out.append("")
    return "\n".join(out).rstrip() + "\n"


@dataclass(frozen=True)
class SummaryLine:
    number: int
    text: str
    term_names: frozenset[str]     # 정규화한 용어 이름


@dataclass
class ParsedSummary:
    lines: list[SummaryLine]

    def by_number(self) -> dict[int, SummaryLine]:
        return {line.number: line for line in self.lines}

    def lines_for(self, term_name: str) -> list[SummaryLine]:
        key = normalize(term_name)
        return [line for line in self.lines if key in line.term_names]

    def format(self) -> str:
        return "\n".join(f"L{line.number}: {line.text}" for line in self.lines)


def parse_summary(markdown: str, term_names: list[str]) -> ParsedSummary:
    """제목(#)이 아닌 내용 줄에 번호를 붙인다. 줄이 속한 용어:
    - '###' 소제목 아래 줄은 그 소제목의 용어
    - 줄 안에 용어 이름이 나오면 그 용어도 (개념 간 관계 줄 등). 짝 개념은 한쪽 이름만 나와도
    """
    keys = {normalize(n): n for n in term_names if normalize(n)}
    # 짝 개념('전역 최솟값·지역 최솟값')은 한쪽 이름만 나온 줄도 그 용어 줄로 본다
    aliases = {k: {k, *(normalize(p) for p in pair_parts(n))} for k, n in keys.items()}
    lines: list[SummaryLine] = []
    section: frozenset[str] = frozenset()
    number = 0
    for raw in markdown.splitlines():
        text = raw.strip()
        if not text:
            continue
        if text.startswith("#"):
            heading = normalize(text.lstrip("#"))
            level = len(text) - len(text.lstrip("#"))
            matched = frozenset(k for k in keys if any(a == heading or a in heading for a in aliases[k]))
            section = matched if level >= 3 else frozenset()
            continue
        number += 1
        body = normalize(text)
        inline = frozenset(k for k in keys if any(a in body for a in aliases[k]))
        lines.append(SummaryLine(number=number, text=text, term_names=section | inline))
    return ParsedSummary(lines=lines)
