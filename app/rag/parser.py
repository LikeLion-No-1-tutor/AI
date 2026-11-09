"""업로드 자료 → 쪽(Page) 목록.

- PDF: PyMuPDF로 쪽마다 텍스트 추출 (PPT·DOCX는 백엔드가 PDF로 바꿔서 보냄)
- 텍스트(.txt .md, 직접 입력): '<!-- page -->' 줄이나 폼피드(\\f)로 쪽을 나눔
- 소스코드: 파일 이름을 머리에 붙이고 120줄씩 한 쪽으로
- 이미지: 여기서는 판별만 한다. 글자·내용 읽기는 services/material.py가 AI로 처리한다.

여러 파일을 함께 올리면 쪽 번호(page)를 1부터 이어서 붙이고,
SourcePage로 '원래 어느 파일의 몇 쪽인지'를 함께 돌려준다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePath

import pymupdf

from ..context import remaining_seconds
from ..errors import LLMTimeout, unreadable, validation_error
from ..schemas.enums import SourceKind

_PAGE_BREAK = re.compile(r"^\s*<!--\s*page\s*-->\s*$", re.IGNORECASE | re.MULTILINE)
MIN_TEXT_CHARS = 50
CODE_LINES_PER_PAGE = 120

TEXT_EXT = {".txt", ".md", ".markdown"}
IMAGE_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}   # 명세: PNG·JPG
# 파일 앞 바이트로 실제 형식을 본다 (확장자만 믿으면 이름만 .png인 파일이 공급자에서 400 → 502가 된다)
_IMAGE_MAGIC = ((b"\x89PNG\r\n\x1a\n", "image/png"), (b"\xff\xd8\xff", "image/jpeg"))
# LLM에 보내는 이미지의 긴 변 상한(px). Claude는 이보다 큰 이미지를 어차피 줄여서 보므로 미리 줄여 보낸다.
# 비정상적으로 큰 PDF 쪽을 렌더링할 때 메모리가 크게 늘어나는 것도 이 값으로 막는다.
MAX_IMAGE_PX = 1568
CODE_EXT = {
    ".py", ".ipynb", ".java", ".c", ".h", ".cpp", ".hpp", ".cc", ".cs", ".js", ".jsx", ".ts", ".tsx",
    ".go", ".rs", ".kt", ".kts", ".swift", ".rb", ".php", ".scala", ".r", ".m", ".sql", ".sh",
    ".html", ".css", ".scss", ".json", ".yaml", ".yml", ".xml", ".dart", ".lua", ".pl",
}
CONVERT_EXT = {".ppt", ".pptx", ".doc", ".docx", ".hwp", ".hwpx", ".key", ".odt", ".odp"}


@dataclass(frozen=True)
class RawPage:
    file_name: str
    kind: SourceKind
    file_page: int
    text: str


@dataclass(frozen=True)
class Page:
    number: int          # 합쳐진 자료 전체에서의 쪽 번호 (1부터)
    file_name: str
    kind: SourceKind
    file_page: int
    text: str


def clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _ext(file_name: str) -> str:
    """로그·에러 원인에는 파일 이름 대신 확장자만 (로그 규칙)."""
    return PurePath(file_name or "").suffix.lower() or "(확장자 없음)"


def _sniff_image(data: bytes) -> str | None:
    for magic, mime in _IMAGE_MAGIC:
        if data.startswith(magic):
            return mime
    return None


def detect_kind(file_name: str, data: bytes) -> SourceKind:
    ext = PurePath(file_name or "").suffix.lower()
    if ext == ".pdf" or data[:5] == b"%PDF-":
        return SourceKind.PDF
    if ext in IMAGE_MIME or _sniff_image(data):
        return SourceKind.IMAGE
    if ext in TEXT_EXT:
        return SourceKind.TEXT
    if ext in CODE_EXT:
        return SourceKind.CODE
    if ext in CONVERT_EXT:
        raise validation_error(f"{_ext(file_name)}: PPT·DOCX 등은 Django에서 PDF로 바꿔서 보내야 해요")
    raise validation_error(f"{_ext(file_name)}: 지원하지 않는 파일 형식")


def prepare_image(file_name: str, data: bytes) -> tuple[bytes, str]:
    """LLM에 보낼 (이미지 바이트, MIME). 형식은 앞 바이트로 판별하고, 긴 변이 MAX_IMAGE_PX를 넘으면 줄인다."""
    mime = _sniff_image(data)
    if mime is None:
        raise unreadable(f"{_ext(file_name)}: PNG·JPG 이미지가 아니거나 손상된 파일")
    try:
        with pymupdf.open(stream=data, filetype=mime.split("/")[1]) as doc:
            rect = doc[0].rect
            longest = max(rect.width, rect.height)
            pix = pymupdf.Pixmap(data)   # 실제 픽셀 크기 (rect는 이미지 DPI에 따라 pt 단위라 다를 수 있음)
            if max(pix.width, pix.height) <= MAX_IMAGE_PX:
                return data, mime
            zoom = MAX_IMAGE_PX / longest
            pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
            if mime == "image/jpeg":
                return pix.tobytes("jpeg", jpg_quality=90), mime
            return pix.tobytes("png"), mime
    except Exception as exc:
        raise unreadable(f"{_ext(file_name)}: 이미지를 열 수 없음") from exc


def _decode(file_name: str, data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp949"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise unreadable(f"{_ext(file_name)}: 텍스트 인코딩을 읽을 수 없음")


def parse_pdf(file_name: str, data: bytes, max_pages: int) -> list[RawPage]:
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise unreadable(f"{_ext(file_name)}: PDF를 열 수 없음") from exc
    with doc:
        if doc.page_count > max_pages:
            raise validation_error(f"{_ext(file_name)}: PDF는 {max_pages}쪽 이하 (현재 {doc.page_count}쪽)")
        return [RawPage(file_name, SourceKind.PDF, i + 1, clean(p.get_text("text"))) for i, p in enumerate(doc)]


def parse_text(file_name: str, text: str) -> list[RawPage]:
    parts = re.split(r"\f|" + _PAGE_BREAK.pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    parts = [clean(p) for p in parts if p.strip()]
    return [RawPage(file_name, SourceKind.TEXT, i + 1, p) for i, p in enumerate(parts)]


def parse_code(file_name: str, text: str) -> list[RawPage]:
    lines = text.replace("\r\n", "\n").split("\n")
    pages = []
    for i in range(0, max(len(lines), 1), CODE_LINES_PER_PAGE):
        body = "\n".join(lines[i : i + CODE_LINES_PER_PAGE]).rstrip()
        if body.strip():
            start, end = i + 1, min(i + CODE_LINES_PER_PAGE, len(lines))
            pages.append(RawPage(file_name, SourceKind.CODE, len(pages) + 1,
                                 f"[소스코드 {file_name} {start}~{end}줄]\n{body}"))
    return pages


def parse_non_image(file_name: str, data: bytes, kind: SourceKind, max_pdf_pages: int) -> list[RawPage]:
    if kind == SourceKind.PDF:
        return parse_pdf(file_name, data, max_pdf_pages)
    text = _decode(file_name, data)
    return parse_code(file_name, text) if kind == SourceKind.CODE else parse_text(file_name, text)


def number_pages(raw: list[RawPage]) -> list[Page]:
    return [Page(i + 1, r.file_name, r.kind, r.file_page, r.text) for i, r in enumerate(raw)]


# ───────────────────────── PDF 쪽 상태 판단 ─────────────────────────
# 쪽마다 글자를 먼저 뽑고, 상태에 따라 읽는 방법을 다르게 한다 (판단은 코드, LLM 호출 없음).
#   text    : 글자만 쓴다
#   figure  : 글자는 그대로 쓰고, 그림 내용만 AI가 이미지를 보고 덧붙인다
#   garbled : 글자가 깨졌거나(글꼴에 글자 정보가 없는 PDF) 거의 없음(스캔본) → 쪽 전체를 AI가 이미지로 읽는다

# 한국어 문장에서 가장 자주 나오는 글자. 정상 문서는 한글의 약 45~55%가 이 글자이고, 글꼴이 깨진 PDF는 0~20%.
_COMMON_HANGUL = set("이다는의에을가하고한로서기지를은리사도수그게있것들으적대자어해인나되정시과일만전주부상와아요까니면라제보장성")
GARBLED_MIN_HANGUL = 20
GARBLED_MAX_COMMON_RATIO = 0.2
SCANNED_MAX_CHARS = 30           # 쪽에 글자가 이보다 적고 그림이 있으면 스캔본으로 본다
_PAGE_NUMBER = re.compile(r"^\s*(?:-\s*)?\d{1,4}(?:\s*/\s*\d{1,4})?(?:\s*-)?\s*$")


@dataclass
class PdfPage:
    file_page: int
    text: str
    mode: str = "text"           # text / figure / garbled
    image: bytes | None = None   # figure·garbled일 때만, PNG


GARBLED_LINE_MIN_HANGUL = 6


# 겹받침(ㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ) 글자는 보통 문장에서 드물다(줄당 20% 미만). 깨진 글꼴에서는 '픡숡먡퐡'처럼 흔하다.
_COMPLEX_FINALS = {3, 5, 6, 9, 10, 11, 12, 13, 14, 15, 18}


def _common_ratio(text: str) -> tuple[int, float]:
    hangul = [ch for ch in text if "가" <= ch <= "힣"]
    if not hangul:
        return 0, 1.0
    return len(hangul), sum(ch in _COMMON_HANGUL for ch in hangul) / len(hangul)


def _complex_ratio(text: str) -> float:
    hangul = [ch for ch in text if "가" <= ch <= "힣"]
    if not hangul:
        return 0.0
    return sum((ord(ch) - 0xAC00) % 28 in _COMPLEX_FINALS for ch in hangul) / len(hangul)


def hangul_is_garbled(text: str) -> bool:
    """쪽 전체가 깨졌거나, 한글 6자 이상인 줄 중 하나라도 깨졌으면 True (글꼴 일부만 깨진 경우도 잡는다)."""
    count, ratio = _common_ratio(text)
    if count >= GARBLED_MIN_HANGUL and ratio < GARBLED_MAX_COMMON_RATIO:
        return True
    for line in text.splitlines():
        n, r = _common_ratio(line)
        if n < GARBLED_LINE_MIN_HANGUL:
            continue
        c = _complex_ratio(line)
        if c >= 0.3 or (r == 0.0 and c >= 0.1):
            return True
    return False


EDGE_RATIO = 0.15   # 쪽 위·아래 15% 안에 있는 줄만 머리말·꼬리말 후보


def _page_lines(page) -> list[tuple[str, bool]]:
    """(줄 글자, 쪽 위·아래 가장자리에 있는지). get_text('text')와 같은 순서."""
    height = page.rect.height or 1
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"])
            if not text.strip():
                continue
            y = (line["bbox"][1] + line["bbox"][3]) / 2 / height
            out.append((text, y < EDGE_RATIO or y > 1 - EDGE_RATIO))
    return out


def _repeated_lines(pages_lines: list[list[tuple[str, bool]]]) -> set[str]:
    """절반 이상의 쪽에서 위·아래 가장자리에 똑같이 나오는 줄 = 머리말·꼬리말·메뉴 줄 (3쪽 이상인 문서만).
    본문에서 반복되는 줄은 지우지 않는다."""
    if len(pages_lines) < 3:
        return set()
    counts: dict[str, int] = {}
    for lines in pages_lines:
        for key in {text.strip() for text, edge in lines if edge}:
            counts[key] = counts.get(key, 0) + 1
    return {k for k, n in counts.items() if n >= max(3, len(pages_lines) / 2)}


def _drop_figure_labels(lines: list[str]) -> list[str]:
    """그림이 있는 쪽: 그림 안 글자 조각(짧은 줄, 같은 쪽에서 반복되는 줄)은 뺀다. 그림 내용은 AI 설명으로 대신한다."""
    seen: dict[str, int] = {}
    for ln in lines:
        seen[ln.strip()] = seen.get(ln.strip(), 0) + 1
    keep = []
    for ln in lines:
        compact = re.sub(r"\s+", "", ln)
        if len(compact) <= 6 or seen[ln.strip()] >= 2:
            continue
        keep.append(ln)
    return keep


def _check_budget(file_name: str) -> None:
    left = remaining_seconds()
    if left is not None and left <= 0:
        raise LLMTimeout(f"{_ext(file_name)}: PDF 분석이 시간 예산을 넘음")


def _render(page: "pymupdf.Page", dpi: int) -> bytes:
    """쪽을 PNG로. 쪽 크기가 비정상적으로 커도 긴 변이 MAX_IMAGE_PX를 넘지 않게 배율을 줄인다."""
    longest_pt = max(page.rect.width, page.rect.height, 1.0)
    zoom = min(dpi / 72, MAX_IMAGE_PX / longest_pt)
    return page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False).tobytes("png")


def analyze_pdf(file_name: str, data: bytes, max_pages: int, dpi: int) -> list[PdfPage]:
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise unreadable(f"{_ext(file_name)}: PDF를 열 수 없음") from exc
    with doc:
        if doc.page_count > max_pages:
            raise validation_error(f"{_ext(file_name)}: PDF는 {max_pages}쪽 이하 (현재 {doc.page_count}쪽)")
        if doc.page_count == 0:
            raise unreadable(f"{_ext(file_name)}: 쪽이 없음")
        page_lines, drawings, image_xrefs = [], [], []
        for p in doc:
            _check_budget(file_name)   # 도형 분석·렌더링 CPU 시간도 요청 시간 예산 안에서
            page_lines.append(_page_lines(p))
            drawings.append(len(p.get_drawings()))
            image_xrefs.append({img[0] for img in p.get_images(full=True)})

        # 여러 쪽에 반복되는 이미지(로고·배경)와 도형(메뉴 막대 등)은 그림으로 치지 않는다
        xref_count: dict[int, int] = {}
        for xs in image_xrefs:
            for x in xs:
                xref_count[x] = xref_count.get(x, 0) + 1
        common_xrefs = {x for x, n in xref_count.items() if doc.page_count >= 3 and n >= doc.page_count / 2}
        base = sorted(drawings)[len(drawings) // 2]               # 보통 쪽의 도형 수 (중앙값)
        figure_threshold = max(base * 1.8, base + 40)

        repeated = _repeated_lines(page_lines)
        pages: list[PdfPage] = []
        for i, page in enumerate(doc):
            _check_budget(file_name)
            lines = [ln for ln, edge in page_lines[i]
                     if not (edge and (ln.strip() in repeated or _PAGE_NUMBER.match(ln)))]
            has_figure = bool(image_xrefs[i] - common_xrefs) or drawings[i] >= figure_threshold
            body = "\n".join(lines)
            if hangul_is_garbled(body) or (len(body.strip()) < SCANNED_MAX_CHARS and has_figure):
                mode = "garbled"
            elif has_figure:
                mode, lines = "figure", _drop_figure_labels(lines)
            else:
                mode = "text"
            text = clean("\n".join(lines)) if mode != "garbled" else ""
            image = _render(page, dpi) if mode != "text" else None
            pages.append(PdfPage(file_page=i + 1, text=text, mode=mode, image=image))
        return pages
