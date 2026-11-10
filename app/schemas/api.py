"""API 명세(AI-0 ~ AI-9)의 요청·응답 구조. 필드 이름은 명세와 똑같이 쓴다.

ID는 Django DB의 ID를 그대로 받고, 결과에도 그대로 돌려준다.
"""
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from .enums import ExamKind, MessageKind, QuestionType, Sender, TermStatus

STUDENT_KINDS = {MessageKind.GREETING, MessageKind.QUESTION}

# 입력 크기 상한 (비용·메모리 보호). 명세의 백엔드 제한(설명 3,000자 등)보다 넉넉하게 둔다.
MAX_TEXT = 10_000          # 메시지 하나, 문제·답 하나
MAX_MESSAGES = 400         # 수업 대화 메시지 수
MAX_TERMS = 50
MAX_QUESTIONS = 30
MAX_SUMMARY = 200_000      # 요약정리 Markdown
MAX_NAME = 200             # 이름·제목·코드·학년 표시
MAX_PROMPT = 20_000        # 페르소나 system_prompt
MAX_CHOICES = 10
Text = Annotated[str, Field(max_length=MAX_TEXT)]
Name = Annotated[str, Field(max_length=MAX_NAME)]
TEACHER_KINDS = {MessageKind.EXPLANATION, MessageKind.ANSWER}


# ───────────────────────── 공통 객체 ─────────────────────────


class Usage(BaseModel):
    role: str          # material, judge, student, evaluator, report
    model: str
    input_tokens: int
    output_tokens: int


class Persona(BaseModel):
    code: Name
    name: Name
    grade_label: Name
    system_prompt: str = Field(max_length=MAX_PROMPT)


class PersonaBrief(BaseModel):
    """AI-9 보고서용 (학생 역할이 아니므로 system_prompt 없음)."""
    name: Name
    grade_label: Name


class Message(BaseModel):
    id: int | None = None          # AI-3의 '방금 보낸 메시지'는 아직 저장 전이라 null
    sender: Sender
    kind: MessageKind
    content: Text
    reply_to_id: int | None = None

    @model_validator(mode="after")
    def _sender_matches_kind(self) -> "Message":
        allowed = STUDENT_KINDS if self.sender == Sender.STUDENT else TEACHER_KINDS
        if self.kind not in allowed:
            raise ValueError(f"sender={self.sender.value}에는 kind={self.kind.value}를 쓸 수 없어요.")
        return self


class TermBrief(BaseModel):
    """AI-1 응답, AI-2 요청의 용어 (아직 DB ID 없음)."""
    name: Name
    definition: Text


# ───────────────────────── AI-1, AI-2 자료 정리 ─────────────────────────


class TermsResponse(BaseModel):
    terms: list[TermBrief]
    usage: list[Usage]


class SummaryResponse(BaseModel):
    summary: str
    usage: list[Usage]


# ───────────────────────── AI-3 용어 판정 + 손 들기 ─────────────────────────


class JudgeMessage(BaseModel):
    kind: MessageKind
    content: Text

    @model_validator(mode="after")
    def _teacher_kind(self) -> "JudgeMessage":
        if self.kind not in TEACHER_KINDS:
            raise ValueError("judge.message.kind는 explanation 또는 answer여야 해요.")
        return self


class JudgeTerm(BaseModel):
    term_id: int
    name: Name
    definition: Text
    status: TermStatus


class JudgePart(BaseModel):
    message: JudgeMessage
    reply_to_question: Text | None = None
    terms: list[JudgeTerm] = Field(default_factory=list, max_length=MAX_TERMS)


class StudentPart(BaseModel):
    persona: Persona
    messages: list[Message] = Field(max_length=MAX_MESSAGES)
    raised_reasons: list[Text] = Field(default_factory=list, max_length=MAX_MESSAGES)


class AnalyzeRequest(BaseModel):
    judge: JudgePart
    student: StudentPart


class TermUpdate(BaseModel):
    term_id: int
    status: TermStatus


class HandRaise(BaseModel):
    reason: str


class AnalyzeResponse(BaseModel):
    term_updates: list[TermUpdate]
    hand_raise: HandRaise | None
    # 학생 반응: 선생님이 질문에 답했고 학생이 다시 손 들지 않았을 때만, 페르소나별 고정 문장 (LLM 호출 없음)
    reaction: str | None = None
    usage: list[Usage]


# ───────────────────────── AI-4 질문 만들기 ─────────────────────────


class QuestionRequest(BaseModel):
    persona: Persona
    messages: list[Message] = Field(max_length=MAX_MESSAGES)
    reason: Text


class QuestionResponse(BaseModel):
    content: str
    usage: list[Usage]


# ───────────────────────── AI-5 정리노트 ─────────────────────────


class NoteRequest(BaseModel):
    title: Name
    persona: Persona
    messages: list[Message] = Field(max_length=MAX_MESSAGES)
    # 손 들었지만 선생님이 끝까지 누르지 않은 이유 (AI-3의 raised_reasons와 같은 값) → '아직 궁금한 사항'
    raised_reasons: list[Text] = Field(default_factory=list, max_length=MAX_MESSAGES)


class NoteResponse(BaseModel):
    content: str
    usage: list[Usage]


# ───────────────────────── AI-6 출제 ─────────────────────────


class ExamTerm(BaseModel):
    term_id: int
    name: Name


class GenerateRequest(BaseModel):
    kind: ExamKind
    summary: str = Field(max_length=MAX_SUMMARY)
    terms: list[ExamTerm] = Field(max_length=MAX_TERMS)
    count: int = Field(ge=1, le=30)
    question_types: list[QuestionType] = Field(min_length=1, max_length=len(QuestionType))
    focus_term_ids: list[int] = Field(default_factory=list, max_length=MAX_TERMS)
    exclude_questions: list[Text] = Field(default_factory=list, max_length=200)


class GeneratedQuestion(BaseModel):
    number: int
    question_type: QuestionType
    content: str
    choices: list[str] | None
    correct_answer: str
    explanation: str
    term_id: int | None


class GenerateResponse(BaseModel):
    questions: list[GeneratedQuestion]
    usage: list[Usage]


# ───────────────────────── AI-7 학생 AI 문제 풀이 ─────────────────────────


class SolveQuestion(BaseModel):
    number: int
    question_type: QuestionType
    content: Text
    choices: list[Text] | None = Field(default=None, max_length=MAX_CHOICES)


class SolveRequest(BaseModel):
    persona: Persona
    messages: list[Message] = Field(max_length=MAX_MESSAGES)
    questions: list[SolveQuestion] = Field(min_length=1, max_length=MAX_QUESTIONS)


class SolveAnswer(BaseModel):
    number: int
    response: str


class SolveResponse(BaseModel):
    answers: list[SolveAnswer]
    usage: list[Usage]


# ───────────────────────── AI-8 주관식 채점 ─────────────────────────


class GradeItem(BaseModel):
    number: int
    content: Text
    correct_answer: Text
    explanation: Text
    response: Text | None = None


class GradeRequest(BaseModel):
    items: list[GradeItem] = Field(min_length=1, max_length=MAX_QUESTIONS)


class GradeResult(BaseModel):
    number: int
    is_correct: bool
    reason: str


class GradeResponse(BaseModel):
    results: list[GradeResult]
    usage: list[Usage]


# ───────────────────────── AI-9 성장보고서 ─────────────────────────


class ReportTerm(BaseModel):
    term_id: int
    name: Name
    definition: Text
    status: TermStatus


class ReportQuestion(BaseModel):
    number: int
    content: Text
    correct_answer: Text
    response: Text | None = None
    is_correct: bool | None = None
    term_id: int | None = None


class ReportExam(BaseModel):
    correct: int
    total: int
    questions: list[ReportQuestion] = Field(max_length=MAX_QUESTIONS)


class ReportRequest(BaseModel):
    title: Name
    persona: PersonaBrief
    messages: list[Message] = Field(max_length=MAX_MESSAGES)
    terms: list[ReportTerm] = Field(max_length=MAX_TERMS)
    exam: ReportExam


class Recommendation(BaseModel):
    term_id: int
    text: str


class ReportResponse(BaseModel):
    strengths: list[str]
    review_recommendations: list[Recommendation]
    usage: list[Usage]
