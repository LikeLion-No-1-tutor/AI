"""티칭로그 AI 서버 (FastAPI). Django만 부른다. DB를 쓰지 않는다.

실행: uvicorn app.main:app --host 0.0.0.0 --port 8001 --no-access-log
문서: /docs

| ID   | Method | 주소                          | 예산(LLM) |
| AI-0 | GET    | /health                       | -         |
| AI-1 | POST   | /materials/terms              | 170초     |
| AI-2 | POST   | /materials/summary            | 170초     |
| AI-3 | POST   | /lessons/messages/analyze     | 25초      |
| AI-4 | POST   | /lessons/questions            | 25초      |
| AI-5 | POST   | /lessons/note                 | 170초     |
| AI-6 | POST   | /exams/generate               | 170초     |
| AI-7 | POST   | /exams/solve                  | 170초     |
| AI-8 | POST   | /exams/grade                  | 문항 1개 25초, 여러 개 170초 |
| AI-9 | POST   | /reports/generate             | 170초     |
"""
from __future__ import annotations

import hmac
import json
import logging
import re
import time
import uuid

from fastapi import Depends, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import TypeAdapter, ValidationError

from . import context
from .config import get_settings
from .errors import AppError, invalid_token, validation_error
from .llm.router import OutputRejected
from .logging_setup import log_event, log_exception, setup_logging
from .schemas.api import (
    AnalyzeRequest,
    AnalyzeResponse,
    GenerateRequest,
    GenerateResponse,
    GradeRequest,
    GradeResponse,
    NoteRequest,
    NoteResponse,
    QuestionRequest,
    QuestionResponse,
    ReportRequest,
    ReportResponse,
    SolveRequest,
    SolveResponse,
    SummaryResponse,
    TermBrief,
    TermsResponse,
)
from .services import exam, grading, lesson, material, note, report

settings = get_settings()
setup_logging(settings.app_env)

# prod에서는 API 문서(/docs, /redoc, /openapi.json)를 열지 않는다
_docs = {} if settings.app_env != "prod" else {"docs_url": None, "redoc_url": None, "openapi_url": None}
app = FastAPI(title="티칭로그 AI 서버", version="2.0.0", **_docs,
              description="AI 학생을 가르치며 배우는 학습 서비스 — AI 서버 (Django 전용)")


# ───────────────────────── 요청 ID · 요청 로그 ─────────────────────────


_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")


def _clean_request_id(raw: str | None) -> str:
    """받은 X-Request-ID는 128자 이하 영숫자·._:- 만 그대로 쓴다. 아니면 새로 만든다 (로그·헤더 오염 방지)."""
    if raw and _REQUEST_ID.fullmatch(raw):
        return raw
    return str(uuid.uuid4())


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = _clean_request_id(request.headers.get("x-request-id"))
    request.state.request_id = request_id
    request.state.error_code = None
    request.state.error_detail = ""
    context.request_id_var.set(request_id)
    started = time.monotonic()
    try:
        response = await call_next(request)
    except Exception as exc:  # 처리하지 못한 예외
        log_exception(exc)
        response = _error_response(request_id, "LLM_ERROR", "LLM 응답을 처리하지 못했어요.", 502)
        request.state.error_code = "LLM_ERROR"
    response.headers["X-Request-ID"] = request_id
    status = response.status_code
    level = logging.ERROR if status >= 500 else logging.WARNING if status >= 400 and status != 401 else logging.INFO
    fields = {"method": request.method, "path": request.url.path, "status": status,
              "duration_ms": int((time.monotonic() - started) * 1000)}
    if request.state.error_code:
        fields["error_code"] = request.state.error_code
    log_event("http_request", level, message=getattr(request.state, "error_detail", "") or "", **fields)
    return response


# ───────────────────────── 에러 형식 ─────────────────────────


def _error_response(request_id: str, code: str, message: str, status: int) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, "request_id": request_id}},
                        headers={"X-Request-ID": request_id})


def _rid(request: Request) -> str:
    return getattr(request.state, "request_id", None) or str(uuid.uuid4())


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    request.state.error_code = exc.code
    request.state.error_detail = exc.detail
    return _error_response(_rid(request), exc.code, exc.message, exc.status_code)


@app.exception_handler(OutputRejected)
async def output_rejected_handler(request: Request, exc: OutputRejected) -> JSONResponse:
    request.state.error_code = "LLM_ERROR"
    request.state.error_detail = str(exc)   # 어떤 작업이 몇 번 검증에 떨어졌는지 (응답에는 넣지 않음)
    return _error_response(_rid(request), "LLM_ERROR", "LLM 응답을 처리하지 못했어요.", 502)


@app.exception_handler(RequestValidationError)
async def request_validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    request.state.error_code = "VALIDATION_ERROR"
    return _error_response(_rid(request), "VALIDATION_ERROR", "요청 형식이 올바르지 않아요.", 422)


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    log_exception(exc)
    request.state.error_code = "LLM_ERROR"
    return _error_response(_rid(request), "LLM_ERROR", "LLM 응답을 처리하지 못했어요.", 502)


# ───────────────────────── 인증 ─────────────────────────


def require_token(x_internal_token: str | None = Header(default=None)) -> None:
    expected = get_settings().ai_server_token
    if not expected:
        if get_settings().app_env == "local":
            return  # 로컬 개발: 토큰 미설정이면 생략
        raise invalid_token()
    if not x_internal_token or not hmac.compare_digest(x_internal_token, expected):
        raise invalid_token()


auth = [Depends(require_token)]


def _realtime() -> context.UsageCollector:
    return context.begin(get_settings().realtime_timeout)


def _background() -> context.UsageCollector:
    return context.begin(get_settings().background_timeout)


# ───────────────────────── AI-0 ─────────────────────────


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# ───────────────────────── AI-1, AI-2 자료 정리 ─────────────────────────


def _upload(file: UploadFile | None) -> material.UploadedFile | None:
    if file is None or not file.filename:
        return None
    limit = get_settings().max_upload_mb * 1024 * 1024
    data = file.file.read(limit + 1)   # 한도보다 1바이트만 더 읽어 초과 여부를 본다 (큰 파일을 통째로 메모리에 올리지 않음)
    if len(data) > limit:
        raise validation_error(f"자료는 {get_settings().max_upload_mb}MB 이하")
    return material.UploadedFile(name=file.filename, data=data)


@app.post("/materials/terms", response_model=TermsResponse, dependencies=auth)
def materials_terms(title: str = Form(...), file: UploadFile | None = File(default=None),
                    text: str | None = Form(default=None)) -> TermsResponse:
    usage = _background()
    ctx = material.load_material(title, _upload(file), text)
    terms = material.extract_terms(ctx)
    return TermsResponse(terms=terms, usage=usage.items())


_terms_adapter = TypeAdapter(list[TermBrief])


@app.post("/materials/summary", response_model=SummaryResponse, dependencies=auth)
def materials_summary(title: str = Form(...), terms: str = Form(...), file: UploadFile | None = File(default=None),
                      text: str | None = Form(default=None)) -> SummaryResponse:
    usage = _background()
    try:
        term_list = _terms_adapter.validate_python(json.loads(terms))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise validation_error("terms는 [{name, definition}] JSON 문자열이어야 해요") from exc
    ctx = material.load_material(title, _upload(file), text)
    summary = material.build_summary(ctx, term_list)
    material.release(ctx)   # AI-1에서 읽어 둔 자료 글은 AI-2가 성공하면 바로 지운다
    return SummaryResponse(summary=summary, usage=usage.items())


# ───────────────────────── AI-3, AI-4, AI-5 수업 ─────────────────────────


@app.post("/lessons/messages/analyze", response_model=AnalyzeResponse, dependencies=auth)
def lessons_analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    usage = _realtime()
    updates, hand, reaction = lesson.analyze(req)
    return AnalyzeResponse(term_updates=updates, hand_raise=hand, reaction=reaction, usage=usage.items())


@app.post("/lessons/questions", response_model=QuestionResponse, dependencies=auth)
def lessons_questions(req: QuestionRequest) -> QuestionResponse:
    usage = _realtime()
    content = lesson.make_question(req)
    return QuestionResponse(content=content, usage=usage.items())


@app.post("/lessons/note", response_model=NoteResponse, dependencies=auth)
def lessons_note(req: NoteRequest) -> NoteResponse:
    usage = _background()
    content, terms = note.write_note(req)
    return NoteResponse(content=content, terms=terms, usage=usage.items())


# ───────────────────────── AI-6, AI-7, AI-8 시험 ─────────────────────────


@app.post("/exams/generate", response_model=GenerateResponse, dependencies=auth)
def exams_generate(req: GenerateRequest) -> GenerateResponse:
    usage = _background()
    questions = exam.generate_exam(req)
    return GenerateResponse(questions=questions, usage=usage.items())


@app.post("/exams/solve", response_model=SolveResponse, dependencies=auth)
def exams_solve(req: SolveRequest) -> SolveResponse:
    usage = _background()
    answers = exam.solve_exam(req)
    return SolveResponse(answers=answers, usage=usage.items())


@app.post("/exams/grade", response_model=GradeResponse, dependencies=auth)
def exams_grade(req: GradeRequest) -> GradeResponse:
    # 재시험은 한 문제씩(실시간), 평가는 주관식 전부를 한 번에(백그라운드)
    usage = _realtime() if len(req.items) == 1 else _background()
    results = grading.grade(req)
    return GradeResponse(results=results, usage=usage.items())


# ───────────────────────── AI-9 보고서 ─────────────────────────


@app.post("/reports/generate", response_model=ReportResponse, dependencies=auth)
def reports_generate(req: ReportRequest) -> ReportResponse:
    usage = _background()
    strengths, recommendations = report.write_report(req)
    return ReportResponse(strengths=strengths, review_recommendations=recommendations, usage=usage.items())
