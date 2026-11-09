"""한국어용 BM25 검색 (글자 2-gram 기반).

외부 임베딩 모델 없이 동작한다 → 무료 등급에서도 쓰고, 자료가 외부로 한 번 더 나가지 않는다.
전공 용어(학습률, 경사하강법 등)처럼 '같은 단어'가 중요한 교재 검색에 잘 맞는다.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .chunker import Chunk

_NON_WORD = re.compile(r"[^0-9a-zA-Z가-힣]+")


def _tokens(text: str) -> list[str]:
    compact = _NON_WORD.sub("", text.lower())
    if len(compact) < 2:
        return [compact] if compact else []
    grams = [compact[i : i + 2] for i in range(len(compact) - 1)]
    # 영문 단어는 통째로도 넣어 'loss' 같은 용어 검색을 돕는다
    grams += re.findall(r"[a-z]{3,}", text.lower())
    return grams


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self._tfs = [Counter(_tokens(c.text)) for c in chunks]
        self._lens = [sum(tf.values()) for tf in self._tfs]
        self._avg = (sum(self._lens) / len(self._lens)) if self._lens else 0.0
        df: Counter[str] = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> list[tuple[Chunk, float]]:
        q = Counter(_tokens(query))
        if not q or not self.chunks:
            return []
        scored: list[tuple[Chunk, float]] = []
        for chunk, tf, length in zip(self.chunks, self._tfs, self._lens):
            score = 0.0
            for term in q:
                f = tf.get(term)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * length / (self._avg or 1))
                score += self._idf.get(term, 0.0) * f * (self.k1 + 1) / denom
            if score > min_score:
                scored.append((chunk, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
