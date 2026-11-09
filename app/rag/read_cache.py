"""AI가 이미지로 읽은 자료 글을 잠깐 기억해 두는 메모리 캐시 (AI-1 → AI-2 두 번 읽기 방지).

API 명세상 AI-1과 AI-2는 같은 파일을 따로 받는다. 이미지·깨진 PDF는 AI가 쪽을 읽어야 해서
(30쪽이면 약 13초, 호출 8번) 두 번 읽으면 비용이 두 배이고, 두 번 읽은 글자가 달라
AI-1 정의의 근거가 AI-2에서 안 맞을 수 있다.

- 열쇠: 파일 내용의 SHA-256 (파일 이름은 보지 않음). 같은 파일이면 같은 결과를 쓴다.
- AI가 읽은 쪽이 있을 때만 저장한다 (글자만 뽑는 일반 PDF·텍스트는 다시 해도 빠르고 같다).
- 원본 폐기 원칙: 디스크에 쓰지 않고 메모리에만, TTL_SECONDS 뒤 자동 삭제, AI-2가 성공하면 바로 삭제.
- 서버 재시작·잠듦이면 사라진다. 그때는 다시 읽는다 (동작은 같음, 비용만 듦).
- 프로세스마다 따로다 (--workers N이면 공유되지 않음).
"""
from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass

TTL_SECONDS = 15 * 60     # AI-1이 끝나고 AI-2가 이 안에 오면 재사용 (Django는 바로 이어서 부른다)
MAX_ENTRIES = 16          # 동시에 준비 중인 자료 수보다 넉넉하게. 넘으면 가장 오래된 것부터 버림


@dataclass(frozen=True)
class _Entry:
    pages: tuple
    expires: float


_lock = threading.Lock()
_store: dict[str, _Entry] = {}


def key_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _purge(now: float) -> None:
    for k in [k for k, e in _store.items() if e.expires <= now]:
        del _store[k]


def get(key: str) -> list | None:
    now = time.monotonic()
    with _lock:
        _purge(now)
        entry = _store.get(key)
        return list(entry.pages) if entry else None


def put(key: str, pages: list) -> None:
    now = time.monotonic()
    with _lock:
        _purge(now)
        _store.pop(key, None)
        while len(_store) >= MAX_ENTRIES:
            del _store[next(iter(_store))]   # dict는 넣은 순서 → 가장 오래된 것
        _store[key] = _Entry(tuple(pages), now + TTL_SECONDS)


def forget(key: str | None) -> None:
    if key:
        with _lock:
            _store.pop(key, None)


def clear() -> None:
    with _lock:
        _store.clear()
