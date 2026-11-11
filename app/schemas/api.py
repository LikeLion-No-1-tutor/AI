"""API 명세(AI-0 ~ AI-9)의 요청·응답 구조. 필드 이름은 명세와 똑같이 쓴다.

ID는 Django DB의 ID를 그대로 받고, 결과에도 그대로 돌려준다.
"""
from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field, model_validator

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


def link_answers(messages: list[Message]) -> list[Message]:
    """질문받기로 나온 학생 질문 뒤의 첫 선생님 메시지는 그 질문의 답변으로 본다 (팀 결정 2026-11-11).

    Django는 프론트가 reply_to_id를 보낼 때만 kind=answer로 저장한다. 프론트가 빠뜨리면 답을 했어도
    '설명'이 되어 정리노트의 질문과 답이 비고 학생 반응도 안 나오므로, 여기서 한 번에 바로잡는다.
    - 답을 기다리는 질문이 있으면 다음 선생님 설명을 answer로 바꾸고 reply_to_id를 그 질문으로 단다.
    - 기다리는 질문이 여럿이면 가장 최근 질문부터 (정리노트 짝 짓기와 같은 순서).
    - 이미 answer인 메시지는 그대로 두고, 그 메시지가 답한 질문만 기다리는 목록에서 뺀다.
    """
    pending: list[Message] = []
    out: list[Message] = []
    for m in messages:
        if m.sender == Sender.STUDENT:
            if m.kind == MessageKind.QUESTION:
                pending.append(m)
        elif m.kind == MessageKind.ANSWER:
            target = next((q for q in pending if m.reply_to_id is not None and q.id == m.reply_to_id), None)
            if target is None and pending and m.reply_to_id is None:
                target = pending[-1]
            if target is not None:
                pending.remove(target)
        elif pending and m.content.strip():
            q = pending.pop()
            m = m.model_copy(update={"kind": MessageKind.ANSWER, "reply_to_id": q.id})
        out.append(m)
    return out


Messages = Annotated[list[Message], Field(max_length=MAX_MESSAGES), AfterValidator(link_answers)]


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
    messages: Messages
    raised_reasons: list[Text] = Field(default_factory=list, max_length=MAX_MESSAGES)


class AnalyzeRequest(BaseModel):
    judge: JudgePart
    student: StudentPart

    @model_validator(mode="after")
    def _judge_follows_link(self) -> "AnalyzeRequest":
        """대화 끝의 방금 메시지가 link_answers로 답변이 됐으면, 용어 판정 쪽도 답변으로 맞춘다."""
        msgs = self.student.messages
        last = msgs[-1] if msgs else None
        if (self.judge.message.kind == MessageKind.EXPLANATION and last is not None
                and last.sender == Sender.TEACHER and last.kind == MessageKind.ANSWER
                and last.content.strip() == self.judge.message.content.strip()):
            question = next((m for m in msgs if m.id is not None and m.id == last.reply_to_id), None)
            self.judge.message.kind = MessageKind.ANSWER
            if question is not None and not self.judge.reply_to_question:
                self.judge.reply_to_question = question.content
        return self


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
    messages: Messages
    reason: Text


class QuestionResponse(BaseModel):
    content: str
    usage: list[Usage]


# ───────────────────────── AI-5 정리노트 ─────────────────────────


class NoteRequest(BaseModel):
    title: Name
    persona: Persona
    messages: Messages
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
    messages: Messages
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
    messages: Messages
    terms: list[ReportTerm] = Field(max_length=MAX_TERMS)
    exam: ReportExam


class Recommendation(BaseModel):
    term_id: int
    text: str


class ReportResponse(BaseModel):
    strengths: list[str]
    review_recommendations: list[Recommendation]
    usage: list[Usage]
