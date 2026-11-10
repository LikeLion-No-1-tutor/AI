"""AI-1 용어 체크리스트 생성, AI-2 요약정리 생성 (역할: material).

공통: 받은 파일·텍스트를 메모리에서만 읽는다 (디스크에 쓰지 않음, 원본 폐기 원칙).
  PDF·텍스트·소스코드는 코드로, 이미지는 [심화] AI가 글자를 읽어 텍스트로 바꾼다.
  → 청크로 나누고 BM25 색인 (이 요청 안에서만 쓰고 버림)

AI-1: [심화] 용어 + 정의. 정의마다 자료 원문 인용을 코드로 검증, 중요한 용어부터.
      + 정의 독립 검사: 다른 호출이 '근거 청크로 뒷받침되는가, 이 용어 자체의 정의인가'를 다시 본다
        (ENABLE_DEFINITION_VERIFICATION). 떨어진 용어는 이유를 붙여 추출을 한 번 다시 부른다.
AI-2: [심화, RAG] 용어마다 관련 청크를 검색하고 용어 3개씩 묶어 동시에 요약(문장마다 원문 인용 검증)
      + [심화] 개념 간 관계(원문 인용 검증) → 코드가 Markdown으로 조립.
      + 본문 검사: 다른 호출이 요점·관계 문장이 근거 청크만으로 나오는지 다시 본다 (ENABLE_CLAIM_VERIFICATION).
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..config import get_settings
from ..context import gather, submit
from ..errors import LLMError, LLMTimeout, unreadable, validation_error
from ..grounding import covered_by, numbers_unsupported, mentions_term, normalize, pair_parts, quote_in_any, term_aliases
from ..llm.base import ImageInput
from ..llm.router import Outcome, OutputRejected, Task, call_structured
from ..logging_setup import log_event
from ..prompts import render
from ..rag import read_cache
from ..rag.chunker import Chunk, chunk_pages
from ..rag.parser import (MIN_TEXT_CHARS, PdfPage, RawPage, analyze_pdf, clean, detect_kind, number_pages, prepare_image,
                          parse_non_image, parse_text)
from ..rag.retriever import BM25Retriever
from ..schemas.api import TermBrief
from ..schemas.enums import SourceKind
from ..schemas.llm_outputs import (
    ExtractedPoint,
    ExtractedRelation,
    DefinitionVerificationOutput,
    ExtractedTerm,
    ImageReadOutput,
    PageReadOutput,
    RelationOutput,
    TermExtractionOutput,
    BatchSummaryOutput,
)
from .common import claim_verification_on, format_chunks, rules, verify_claims
from .summary_doc import SummaryTerm, build_summary_markdown

SUMMARY_TOP_K = 6
SUMMARY_TERMS_PER_CALL = 3     # 요약은 용어 3개씩 나눠 동시에 부른다 (한 번에 10개면 출력이 1만 토큰을 넘어 잘림)
MIN_POINTS_KEEP = 2            # 문장을 잃은 용어의 요점이 이보다 적어지면 그 묶음을 다시 부른다
RELATION_CHUNKS_PER_TERM = 3
SHORT_MATERIAL_CHARS = 150     # 이보다 짧은 자료는 최소 개수를 강제하지 않는다 (핵심이 하나뿐일 수 있음)


@dataclass(frozen=True)
class UploadedFile:
    name: str
    data: bytes


@dataclass
class MaterialContext:
    title: str
    chunks: list[Chunk]
    retriever: BM25Retriever
    total_chars: int
    cache_key: str | None = None   # AI가 읽은 쪽을 캐시에 둔 경우 그 열쇠 (AI-2 성공 후 forget)

    def chunk_map(self) -> dict[str, Chunk]:
        return {c.id: c for c in self.chunks}

    def relevant(self, query: str, must_include: list[str], top_k: int) -> list[Chunk]:
        cmap = self.chunk_map()
        picked = {i: cmap[i] for i in must_include if i in cmap}
        for chunk, _ in self.retriever.search(query, top_k=top_k):
            picked.setdefault(chunk.id, chunk)
        return sorted(picked.values(), key=lambda c: (c.page, c.id))


# ───────────────────────── 자료 읽기 ─────────────────────────


def _read_image(file: UploadedFile) -> str:
    data, mime = prepare_image(file.name, file.data)
    out = call_structured(Task.IMAGE_READ, system=render("image_read"),
                          user="이 이미지의 학습 내용을 글로 옮겨주세요.",
                          output_model=ImageReadOutput,
                          images=[ImageInput(data, mime)])
    return clean(out.value.text)


def _read_pdf_pages(pages: list[PdfPage]) -> None:
    """figure·garbled 쪽을 이미지로 AI에게 보여주고 글을 채운다 (여러 쪽씩 묶어 한 번에, 묶음끼리는 동시에).

    garbled: 쪽 전체를 옮긴 글로 바꾼다. figure: 뽑힌 본문 뒤에 [그림: …] 설명을 붙인다.
    한계: 이미지로 읽은 글은 원본 글자와 대조할 수 없다 (잘못 읽은 글자는 검증으로 잡히지 않음).
    """
    s = get_settings()
    targets = [p for p in pages if p.mode != "text" and p.image]
    if not targets:
        return
    size = s.pages_per_read
    batches = [targets[i:i + size] for i in range(0, len(targets), size)]

    def read(batch: list[PdfPage]) -> dict[int, str]:
        wanted = {p.file_page for p in batch}
        listing = []
        for n, p in enumerate(batch, 1):
            if p.mode == "garbled":
                listing.append(f"- 이미지 {n} = {p.file_page}쪽: [전체 읽기]")
            else:
                listing.append(f"- 이미지 {n} = {p.file_page}쪽: [그림만 설명]\n  이미 뽑힌 본문:\n"
                               + "\n".join(f"    {ln}" for ln in (p.text or "(없음)").splitlines()))

        def validate(out: PageReadOutput) -> Outcome[dict[int, str]]:
            got = {x.page: x.text.strip() for x in out.pages if x.page in wanted}
            missing = sorted(wanted - set(got))
            return Outcome(got, fatal=[f"빠진 쪽: {missing}"] if missing else [])

        return call_structured(Task.PAGE_READ, system=render("page_read"),
                               user="[쪽 목록 — 이미지 순서대로]\n" + "\n".join(listing),
                               output_model=PageReadOutput, validate=validate,
                               images=[ImageInput(p.image, "image/png") for p in batch]).value

    with ThreadPoolExecutor(max_workers=max(1, s.llm_concurrency)) as pool:
        results = gather(pool, [(read, (b,)) for b in batches])
    texts = {k: v for r in results for k, v in r.items()}
    for p in targets:
        got = clean(texts.get(p.file_page, ""))
        if p.mode == "garbled":
            p.text = got
        elif got:
            lines = [ln if ln.startswith("[그림") else f"[그림: {ln}]" for ln in got.splitlines() if ln.strip()]
            p.text = clean("\n".join([p.text, *lines]) if p.text else "\n".join(lines))
    log_event("pdf_pages_read", logging.DEBUG, pages=len(pages), figure=sum(p.mode == "figure" for p in pages),
              garbled=sum(p.mode == "garbled" for p in pages), calls=len(batches))


def load_material(title: str, file: UploadedFile | None, text: str | None) -> MaterialContext:
    s = get_settings()
    title = (title or "").strip()
    if not title:
        raise validation_error("title이 비어 있어요")
    text = text or ""
    if file is None and not text.strip():
        raise validation_error("file과 text 중 하나는 있어야 해요")
    size = (len(file.data) if file else 0) + len(text.encode())
    if size > s.max_upload_mb * 1024 * 1024:
        raise validation_error(f"자료는 {s.max_upload_mb}MB 이하")

    raw: list[RawPage] = []
    cache_key: str | None = None
    if file is not None:
        if not file.data:
            raise unreadable("빈 파일")
        kind = detect_kind(file.name, file.data)
        key = read_cache.key_of(file.data)
        cached = read_cache.get(key) if kind in (SourceKind.IMAGE, SourceKind.PDF) else None
        if cached is not None:
            raw.extend(cached)
            cache_key = key
            log_event("material_read_cache_hit", logging.DEBUG, pages=len(cached))
        elif kind == SourceKind.IMAGE:
            if len(file.data) > s.max_image_mb * 1024 * 1024:
                raise validation_error(f"이미지는 {s.max_image_mb}MB 이하")
            image_text = _read_image(file)
            if image_text:
                raw.append(RawPage(file.name, SourceKind.IMAGE, 1, image_text))
            read_cache.put(key, raw)
            cache_key = key
        elif kind == SourceKind.PDF:
            pdf_pages = analyze_pdf(file.name, file.data, s.max_pdf_pages, s.pdf_render_dpi)
            _read_pdf_pages(pdf_pages)
            raw.extend(RawPage(file.name, SourceKind.PDF, p.file_page, p.text) for p in pdf_pages)
            if any(p.mode != "text" and p.image for p in pdf_pages):   # AI가 읽은 쪽이 있을 때만 기억
                read_cache.put(key, raw)
                cache_key = key
        else:
            raw.extend(parse_non_image(file.name, file.data, kind, s.max_pdf_pages))
        log_event("material_loaded", logging.DEBUG, file_ext=(file.name.rsplit(".", 1)[-1].lower() if "." in file.name else ""),
                  file_bytes=len(file.data), kind=kind.value)
    if text.strip():
        raw.extend(parse_text("직접 입력", text))

    raw = [r for r in raw if r.text]
    total_chars = sum(len(r.text) for r in raw)
    if total_chars < MIN_TEXT_CHARS:
        raise unreadable("글자를 찾지 못함")
    pages = number_pages(raw)
    chunks = chunk_pages(pages)
    return MaterialContext(title=title, chunks=chunks, retriever=BM25Retriever(chunks), total_chars=total_chars,
                           cache_key=cache_key)


def release(ctx: MaterialContext) -> None:
    """AI-2가 끝나면 캐시에 둔 자료 글을 바로 지운다 (원본 폐기 원칙)."""
    read_cache.forget(ctx.cache_key)


# ───────────────────────── AI-1 용어 체크리스트 ─────────────────────────


def term_count_range(total_chars: int) -> tuple[int, int]:
    """체크리스트 개수 범위 (최소, 최대). 개수는 자료 길이가 아니라 AI가 고른 핵심 용어 수로 정해진다."""
    s = get_settings()
    return (1 if total_chars < SHORT_MATERIAL_CHARS else s.min_terms), s.max_terms


def extract_terms(ctx: MaterialContext) -> list[TermBrief]:
    min_ok, count = term_count_range(ctx.total_chars)
    cmap = ctx.chunk_map()
    material_text = normalize(" ".join(c.text for c in ctx.chunks))
    verify = get_settings().enable_definition_verification

    verify_cache: dict[tuple, str | None] = {}

    def validate(out: TermExtractionOutput) -> Outcome[list[ExtractedTerm]]:
        kept: list[ExtractedTerm] = []
        fatal: list[str] = []
        seen: set[str] = set()
        for t in out.terms:
            name, definition = t.name.strip(), t.definition.strip()
            key = normalize(name)
            if not key or key in seen:
                continue
            if not definition:
                fatal.append(f"용어 '{name}': 정의가 비어 있습니다.")
                continue
            if not any(alias in material_text for alias in term_aliases(name)):
                # 체크리스트 이름은 선생님이 수업에서 말하는 이름이 된다 → 자료에 그대로 나오는 용어여야 한다
                fatal.append(f"용어 '{name}': 자료에 그대로 나오는 용어 이름이 아닙니다. 설명구('~의 한계', '~ 수식', '~하기') 대신 "
                             "자료에 쓰인 용어 이름을 그대로 쓰세요.")
                continue
            ids = [i for i in t.chunk_ids if i in cmap]
            if not ids:
                fatal.append(f"용어 '{name}': chunk_ids {t.chunk_ids}가 제공된 청크에 없습니다.")
                continue
            if not quote_in_any(t.evidence_quote, [cmap[i].text for i in ids]):
                fatal.append(f"용어 '{name}': evidence_quote가 청크 {ids} 원문에 없습니다. 원문을 글자 그대로 복사하세요.")
                continue
            bad_numbers = numbers_unsupported(definition, [cmap[i].text for i in ids])
            if bad_numbers:
                fatal.append(f"용어 '{name}': 정의에 청크 {ids} 원문에 없는 숫자({', '.join(bad_numbers)})가 있습니다. "
                             "원문에 쓰인 숫자만 쓰세요.")
                continue
            seen.add(key)
            kept.append(t.model_copy(update={"name": name, "definition": definition, "chunk_ids": ids}))
        if verify and kept:
            failed = _verify_definitions(kept, cmap, verify_cache)
            for i, comment in sorted(failed.items()):
                fatal.append(f"용어 '{kept[i].name}': 정의 검사 탈락 — {comment} "
                             "이 용어 자체가 무엇인지 말하는 자료 문장을 근거로 다시 정의하세요.")
            kept = [t for i, t in enumerate(kept) if i not in failed]
        kept = kept[:count]
        if len(kept) < min_ok:
            fatal.append(f"근거가 확인된 용어가 {len(kept)}개뿐입니다. 최소 {min_ok}개의 핵심 용어를 고르세요.")
        return Outcome(kept, fatal=fatal)

    system = render("term_extract", title=ctx.title, min_count=str(min_ok), max_count=str(count), rules=rules())
    user = f"[주제]\n{ctx.title}\n\n[학습 자료 청크]\n{format_chunks(ctx.chunks)}"
    result = call_structured(Task.TERM_EXTRACT, system=system, user=user,
                             output_model=TermExtractionOutput, validate=validate)
    if not result.value:
        raise unreadable("자료에서 근거가 확인되는 용어를 찾지 못함")
    if result.warnings:
        log_event("terms_dropped", logging.DEBUG, dropped=len(result.warnings))
    return [TermBrief(name=t.name, definition=t.definition) for t in result.value]


_NOT_CHECKED = "정의 검사를 받지 못했습니다."


def _verify_definitions(terms: list[ExtractedTerm], cmap: dict[str, Chunk],
                        cache: dict[tuple, str | None] | None = None) -> dict[int, str]:
    """정의 독립 검사 (LLM 한 번). 떨어진 항목의 {번호: 이유}. 검사 응답에서 빠진 항목도 이번엔 떨어뜨린다.

    인용 검증은 '근거 문장이 자료에 있는지'만 본다. 여기서는 정의가 그 근거와 같은 말인지,
    다른 개념·식·예시를 정의로 가져온 것은 아닌지를 본다 (예: '학습률' 정의로 업데이트 식).
    본문 검사(verify_claims)와 같은 규칙:
    - cache {(이름, 정의, 근거 청크): 탈락 이유 또는 None}: 다시 부를 때 이미 본 용어는 검사하지 않는다.
    - 검사 호출 자체가 실패(LLM 오류·시간 초과)하면 새 용어는 통과시킨다 (이미 인용·자료 대조를 통과함).
      통과시킨 용어는 캐시하지 않으므로 다시 부르면 그때 다시 검사한다.
    """
    cache = {} if cache is None else cache
    keys = [(t.name, t.definition, tuple(t.chunk_ids)) for t in terms]
    todo = [i for i, k in enumerate(keys) if k not in cache]
    missing: dict[int, str] = {}
    if todo:
        try:
            failed = _check_definitions([terms[i] for i in todo], cmap)
        except (LLMError, LLMTimeout, OutputRejected) as exc:
            log_event("definition_verification_skipped", logging.DEBUG,
                      message=f"정의 검사 호출 실패 → 새 용어 {len(todo)}개는 코드 검증만으로 통과",
                      error_code=getattr(exc, "code", "LLM_ERROR"))
            failed = {}
        else:
            for k, i in enumerate(todo):
                if failed.get(k) == _NOT_CHECKED:
                    missing[i] = _NOT_CHECKED
                else:
                    cache[keys[i]] = failed.get(k)
    out = {i: cache[k] for i, k in enumerate(keys) if cache.get(k)}
    out.update(missing)
    if out:
        log_event("definitions_rejected", logging.DEBUG, count=len(out))
    return out


def _check_definitions(terms: list[ExtractedTerm], cmap: dict[str, Chunk]) -> dict[int, str]:
    blocks = []
    for i, t in enumerate(terms):
        chunks = "\n".join(f"  [{c}] {cmap[c].text}" for c in t.chunk_ids)
        blocks.append(f"[index {i}] 용어: {t.name}\n  정의: {t.definition}\n  근거 청크:\n{chunks}")
    wanted = set(range(len(terms)))

    def validate(out: DefinitionVerificationOutput) -> Outcome[dict[int, str]]:
        failed: dict[int, str] = {}
        seen: set[int] = set()
        for c in out.checks:
            if c.index not in wanted or c.index in seen:
                continue
            seen.add(c.index)
            if not (c.supported and c.defines_term):
                failed[c.index] = c.comment.strip() or "근거와 맞지 않습니다."
        missing = sorted(wanted - seen)
        for i in missing:
            failed[i] = _NOT_CHECKED
        return Outcome(failed, fatal=[f"검사가 빠진 index: {missing}"] if missing else [])

    return call_structured(Task.TERM_VERIFY, system=render("term_verify"), user="\n\n".join(blocks),
                           output_model=DefinitionVerificationOutput, validate=validate).value


# ───────────────────────── AI-2 요약정리 ─────────────────────────


def _summarize_terms(ctx: MaterialContext, terms: list[TermBrief],
                     all_terms: list[TermBrief] | None = None) -> list[list[tuple[str, list[int]]]]:
    """[RAG] 용어를 SUMMARY_TERMS_PER_CALL개씩 묶어 동시에 요약한다.

    한 번에 모든 용어를 부르면 출력이 너무 길어 max_tokens에서 잘리고, 문장 하나만 떨어져도 전부 다시 만들어야 한다.
    묶음으로 나누면 출력이 짧고, 검증에 떨어진 묶음만 다시 부른다.
    all_terms: '다른 용어에 대한 문장' 판별에 쓰는 전체 용어 목록 (없으면 terms).
    """
    all_terms = all_terms or terms
    batches = [terms[i:i + SUMMARY_TERMS_PER_CALL] for i in range(0, len(terms), SUMMARY_TERMS_PER_CALL)]
    if len(batches) <= 1:
        return _summarize_batch(ctx, terms, all_terms)
    with ThreadPoolExecutor(max_workers=max(1, min(len(batches), get_settings().llm_concurrency))) as pool:
        results = gather(pool, [(_summarize_batch, (ctx, b, all_terms)) for b in batches])
    return [points for r in results for points in r]


def _summarize_batch(ctx: MaterialContext, terms: list[TermBrief],
                     all_terms: list[TermBrief]) -> list[list[tuple[str, list[int]]]]:
    """용어 묶음 하나를 LLM 한 번에 요약한다. 문장마다 '그 용어에 배정된 청크' 안에서만 근거를 인정한다."""
    allowed: list[dict[str, Chunk]] = [
        {c.id: c for c in ctx.relevant(f"{t.name} {t.definition}", [], SUMMARY_TOP_K)} for t in terms
    ]
    pool: dict[str, Chunk] = {}
    for chunk_map in allowed:
        pool.update(chunk_map)
    if not pool:
        return [[] for _ in terms]
    index_by_name = {normalize(t.name): i for i, t in enumerate(terms)}
    # 다른 용어 판별용: 자료의 절반 이상 청크에 나오는 이름(예: 'RNN')은 너무 흔해서 빼고 본다
    common = {t.name for t in all_terms
              if sum(mentions_term(t.name, c.text) for c in ctx.chunks) > len(ctx.chunks) / 2}

    def belongs_elsewhere(i: int, text: str, chunk_texts: list[str]) -> str | None:
        """근거 청크에도 문장에도 이 용어가 없고, 근거 청크가 다른 용어를 다루면 그 용어 이름 (다른 개념의 문장)."""
        own = terms[i].name
        if mentions_term(own, text) or any(mentions_term(own, ct) for ct in chunk_texts):
            return None
        for t in all_terms:
            if t.name != own and t.name not in common and any(mentions_term(t.name, ct) for ct in chunk_texts):
                return t.name
        return None

    verify_cache: dict[str, str | None] = {}

    def validate(out: BatchSummaryOutput) -> Outcome[dict[int, list[ExtractedPoint]]]:
        result: dict[int, list[ExtractedPoint]] = {}
        fatal: list[str] = []
        hit: set[int] = set()          # 문장이 떨어진 용어
        for block in out.terms:
            i = index_by_name.get(normalize(block.term))
            if i is None:
                fatal.append(f"용어 '{block.term}'는 요청한 용어가 아닙니다.")
                continue
            kept = result.setdefault(i, [])
            seen = {normalize(terms[i].definition)} | {normalize(p.text) for p in kept}
            for n, p in enumerate(block.points, 1):
                text = p.text.strip()
                if not text or normalize(text) in seen:
                    continue
                ids = [x for x in p.chunk_ids if x in allowed[i]]
                if not ids:
                    hit.add(i)
                    fatal.append(f"{terms[i].name} {n}번 문장: chunk_ids {p.chunk_ids}가 이 용어의 청크가 아닙니다.")
                    continue
                if not quote_in_any(p.evidence_quote, [allowed[i][x].text for x in ids]):
                    hit.add(i)
                    fatal.append(f"{terms[i].name} {n}번 문장: evidence_quote가 청크 원문에 없습니다.")
                    continue
                bad_numbers = numbers_unsupported(text, [allowed[i][x].text for x in ids])
                if bad_numbers:
                    hit.add(i)
                    fatal.append(f"{terms[i].name} {n}번 문장: 근거 청크에 없는 숫자({', '.join(bad_numbers)})가 있습니다. "
                                 "원문에 쓰인 숫자만 쓰세요.")
                    continue
                other = belongs_elsewhere(i, text, [allowed[i][x].text for x in ids])
                if other:
                    hit.add(i)
                    fatal.append(f"{terms[i].name} {n}번 문장: 근거 청크는 '{other}'에 대한 내용입니다. "
                                 f"'{terms[i].name}' 자체에 대한 문장만 쓰세요.")
                    continue
                seen.add(normalize(text))
                kept.append(p.model_copy(update={"text": text, "chunk_ids": ids}))
        if claim_verification_on():
            flat = [(i, p) for i, ps in result.items() for p in ps]
            items = [f"용어: {terms[i].name}\n문장: {p.text}\n근거 청크: {', '.join(p.chunk_ids)}" for i, p in flat]
            failed = verify_claims(Task.SUMMARY_VERIFY, "summary_verify", f"[근거 청크]\n{format_chunks(chunks)}", items,
                                    verify_cache)
            for k, comment in sorted(failed.items()):
                i, p = flat[k]
                hit.add(i)
                fatal.append(f"{terms[i].name} 문장 '{p.text[:40]}': 검사 탈락 — {comment} 근거 청크에 쓰인 내용만 쓰세요.")
            drop = {id(flat[k][1]) for k in failed}
            result = {i: [p for p in ps if id(p) not in drop] for i, ps in result.items()}
        # 떨어진 문장은 빼고 남은 요점을 쓴다 (PRD 7장: 남은 부분으로 응답). 묶음 전체를 다시 부르는 건
        # 문장을 잃은 용어의 요점이 MIN_POINTS_KEEP개 미만으로 줄었을 때만. (자료가 얇아 처음부터 1개인 용어는 다시 부르지 않음)
        thin = [terms[i].name for i in sorted(hit) if len(result.get(i, [])) < MIN_POINTS_KEEP]
        if not thin:
            return Outcome(result, warnings=fatal)
        fatal.append(f"요점이 {MIN_POINTS_KEEP}개 미만으로 줄어든 용어: {', '.join(thin)}. 근거가 확인되는 요점을 더 쓰세요.")
        return Outcome(result, fatal=fatal)

    blocks = []
    for t, chunk_map in zip(terms, allowed):
        ids = ", ".join(sorted(chunk_map, key=lambda x: (pool[x].page, x))) or "(관련 청크 없음)"
        blocks.append(f"- {t.name}: {t.definition}\n  이 용어에 쓸 수 있는 청크: {ids}")
    chunks = sorted(pool.values(), key=lambda c: (c.page, c.id))
    system = render("term_summary", rules=rules())
    user = (f"[주제] {ctx.title}\n\n[용어와 정의]\n" + "\n".join(blocks) +
            f"\n\n[학습 자료 청크]\n{format_chunks(chunks)}")
    result = call_structured(Task.TERM_SUMMARY, system=system, user=user,
                             output_model=BatchSummaryOutput, validate=validate).value
    return [[(p.text, sorted({pool[c].page for c in p.chunk_ids})) for p in result.get(i, [])]
            for i in range(len(terms))]


def _find_relations(ctx: MaterialContext, terms: list[TermBrief]) -> list[tuple[str, list[int]]]:
    if len(terms) < 2:
        return []
    pool: dict[str, Chunk] = {}
    for t in terms:
        for c in ctx.relevant(t.name, [], RELATION_CHUNKS_PER_TERM):
            pool.setdefault(c.id, c)
    if not pool:
        return []
    chunks = sorted(pool.values(), key=lambda c: (c.page, c.id))
    names = {normalize(t.name): t.name for t in terms}

    def _named_in(key: str, body: str) -> bool:
        # 짝 개념('전역 최솟값·지역 최솟값')은 한쪽 이름만 나와도 된다
        return key in body or any(normalize(p) in body for p in pair_parts(names[key]))
    max_rel = len(terms) * 2
    verify_cache: dict[str, str | None] = {}

    def validate(out: RelationOutput) -> Outcome[list[ExtractedRelation]]:
        kept: list[ExtractedRelation] = []
        fatal: list[str] = []
        pairs: set[frozenset[str]] = set()
        for r in out.relations:
            a, b = normalize(r.from_term), normalize(r.to_term)
            if a not in names or b not in names or a == b:
                fatal.append(f"관계: 용어 이름이 목록에 없습니다 ({r.from_term}, {r.to_term}).")
                continue
            pair = frozenset((a, b))
            body = normalize(r.text)
            if pair in pairs or not r.text.strip() or not _named_in(a, body) or not _named_in(b, body):
                continue   # 같은 쌍 반복, 또는 문장에 두 용어가 다 들어 있지 않은 것(관계가 아니라 한 용어 설명)은 뺀다
            ids = [x for x in r.chunk_ids if x in pool]
            if not ids or not quote_in_any(r.evidence_quote, [pool[x].text for x in ids]):
                fatal.append(f"관계 ({r.from_term}, {r.to_term}): evidence_quote가 청크 원문에 없습니다.")
                continue
            bad_numbers = numbers_unsupported(r.text, [pool[x].text for x in ids])
            if bad_numbers:
                fatal.append(f"관계 ({r.from_term}, {r.to_term}): 근거 청크에 없는 숫자({', '.join(bad_numbers)})가 있습니다. "
                             "원문에 쓰인 숫자만 쓰세요.")
                continue
            pairs.add(pair)
            kept.append(r.model_copy(update={"chunk_ids": ids, "text": r.text.strip()}))
        kept = kept[:max_rel]
        if claim_verification_on() and kept:
            items = [f"관계 문장: {r.text}\n근거 청크: {', '.join(r.chunk_ids)}" for r in kept]
            failed = verify_claims(Task.SUMMARY_VERIFY, "summary_verify", f"[근거 청크]\n{format_chunks(chunks)}", items,
                                    verify_cache)
            for k, comment in sorted(failed.items()):
                fatal.append(f"관계 ({kept[k].from_term}, {kept[k].to_term}): 검사 탈락 — {comment} 근거 청크에 쓰인 관계만 쓰세요.")
            kept = [r for k, r in enumerate(kept) if k not in failed]
        return Outcome(kept, fatal=fatal)

    term_block = "\n".join(f"- {t.name}: {t.definition}" for t in terms)
    system = render("relations", max=str(max_rel), rules=rules())
    user = f"[용어 목록]\n{term_block}\n\n[관련 청크]\n{format_chunks(chunks)}"
    result = call_structured(Task.RELATIONS, system=system, user=user, output_model=RelationOutput, validate=validate)
    return [(r.text, sorted({pool[c].page for c in r.chunk_ids})) for r in result.value]


def _drop_repeats(terms: list[TermBrief], points: list[list[tuple[str, list[int]]]],
                  relations: list[tuple[str, list[int]]]):
    """이미 쓴 말을 되풀이하는 문장을 뺀다 (코드로, LLM 호출 없음).

    - 요약 문장: 그 용어의 정의, 또는 앞 용어에서 이미 쓴 문장과 같은 말이면 뺀다.
    - 관계 문장: 어느 용어의 정의나 요약 문장과 같은 말이면 뺀다.
    """
    written: list[str] = []
    cleaned: list[list[tuple[str, list[int]]]] = []
    for t, term_points in zip(terms, points):
        kept = []
        for text, pages in term_points:
            if covered_by(text, t.definition, ignore=t.name) or any(covered_by(text, w, ignore=t.name) for w in written):
                continue
            kept.append((text, pages))
            written.append(text)
        cleaned.append(kept)
    references = [t.definition for t in terms] + written
    kept_relations = [(text, pages) for text, pages in relations
                      if not any(covered_by(text, ref) for ref in references)]
    dropped = sum(map(len, points)) - sum(map(len, cleaned)) + len(relations) - len(kept_relations)
    if dropped:
        log_event("summary_repeats_dropped", logging.DEBUG, count=dropped)
    return cleaned, kept_relations


def _fill_empty_terms(ctx: MaterialContext, terms: list[TermBrief],
                      points: list[list[tuple[str, list[int]]]]) -> list[list[tuple[str, list[int]]]]:
    """요약 문장이 하나도 없는 용어만 모아 한 번 더 요약한다 (그 용어들만 보내므로 짧은 호출).

    다시 불러도 비거나 실패하면 그대로 둔다: 정의 줄은 항상 있으므로 요약정리 전체를 실패시키지 않는다.
    """
    empty = [i for i, p in enumerate(points) if not p]
    if not empty:
        return points
    try:
        again = _summarize_terms(ctx, [terms[i] for i in empty], terms)
    except (LLMError, LLMTimeout, OutputRejected) as exc:
        log_event("summary_refill_failed", logging.DEBUG, count=len(empty), error_code=getattr(exc, "code", "LLM_ERROR"))
        return points
    filled = list(points)
    for i, extra in zip(empty, again):
        filled[i] = extra
    return filled


def build_summary(ctx: MaterialContext, terms: list[TermBrief]) -> str:
    if not terms:
        raise validation_error("terms가 비어 있어요")
    s = get_settings()
    with ThreadPoolExecutor(max_workers=max(1, min(2, s.llm_concurrency))) as pool:
        points, relations = gather(pool, [(_summarize_terms, (ctx, terms)), (_find_relations, (ctx, terms))])
    points, relations = _drop_repeats(terms, points, relations)   # 되풀이를 먼저 빼야 '요점 0개'인 용어를 제대로 찾는다
    points = _fill_empty_terms(ctx, terms, points)
    points, relations = _drop_repeats(terms, points, relations)   # 다시 채운 요점도 되풀이 검사
    summary_terms = [SummaryTerm(name=t.name, definition=t.definition, points=p) for t, p in zip(terms, points)]
    empty = sum(1 for p in points if not p)
    if empty:
        log_event("summary_terms_without_points", logging.DEBUG, count=empty)
    return build_summary_markdown(ctx.title, summary_terms, relations)
