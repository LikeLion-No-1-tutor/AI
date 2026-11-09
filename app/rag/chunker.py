"""쪽 텍스트를 검색 단위(청크)로 자른다. 청크 ID 형식: p{쪽}-c{순번}

청크는 자료 처리 요청 안에서만 쓰이고, 응답 후 원본과 함께 사라진다.
"""
from __future__ import annotations

from dataclasses import dataclass

from .parser import Page


@dataclass(frozen=True)
class Chunk:
    id: str
    page: int
    text: str


def chunk_pages(pages: list[Page], size: int = 600, overlap: int = 120) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("overlap은 size보다 작아야 합니다")
    chunks: list[Chunk] = []
    for page in pages:
        text = page.text
        if not text:
            continue
        # 문단 경계를 최대한 살려 자른다
        start, idx = 0, 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                cut = max(text.rfind("\n", start + size // 2, end), text.rfind(". ", start + size // 2, end))
                if cut > start:
                    end = cut + 1
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(id=f"p{page.number}-c{idx}", page=page.number, text=piece))
                idx += 1
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks
