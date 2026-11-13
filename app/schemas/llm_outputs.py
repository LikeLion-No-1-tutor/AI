"""LLM이 반드시 이 형태로만 답하게 하는 출력 스키마 (API 응답과는 별개인 내부 형식).

- 모델에 JSON Schema로 형식을 강제한다 (Claude: output_config.format, Gemini: response_json_schema).
- 받은 뒤 Pydantic(Enum 포함)으로 다시 검증하고, 근거(인용·번호)는 services에서 코드로 검사한다.
- 근거를 검사할 수 있도록 API 응답에는 없는 evidence_quote, message_refs 등을 함께 받는다.
- 필드 이름에 reasoning/thinking 같은 단어는 쓰지 않는다 (Claude가 추론 노출 요청으로 보고 거절할 수 있음).
"""
from __future__ import annotations

from pydantic import BaseModel

from .enums import QuestionType, TermStatus

# ── 자료 정리 (AI-1, AI-2) ──


class ImageReadOutput(BaseModel):
    text: str


class PageText(BaseModel):
    page: int
    text: str


class PageReadOutput(BaseModel):
    pages: list[PageText]


class ExtractedTerm(BaseModel):
    name: str
    definition: str
    chunk_ids: list[str]
    evidence_quote: str


class DefinitionCheck(BaseModel):
    # comment를 결론보다 먼저 둔다: 모델은 필드 순서대로 쓰므로, 따져 본 뒤에 결론을 내게 된다
    index: int
    comment: str
    supported: bool          # 정의의 내용이 근거 청크에 쓰여 있는가
    defines_term: bool       # 다른 개념·식·예시가 아니라 이 용어 자체의 정의인가


class DefinitionVerificationOutput(BaseModel):
    checks: list[DefinitionCheck]


class ClaimCheck(BaseModel):
    # comment를 결론보다 먼저 둔다 (결론부터 쓰고 '실제로는 문제 없음'이라고 뒤집는 일을 막는다)
    index: int
    comment: str
    supported: bool          # 이 문장이 주어진 근거만으로 나오는가


class ClaimVerificationOutput(BaseModel):
    checks: list[ClaimCheck]


class TermExtractionOutput(BaseModel):
    terms: list[ExtractedTerm]


class ExtractedPoint(BaseModel):
    text: str
    chunk_ids: list[str]
    evidence_quote: str


class TermPoints(BaseModel):
    term: str
    points: list[ExtractedPoint]


class BatchSummaryOutput(BaseModel):
    terms: list[TermPoints]


class ExtractedRelation(BaseModel):
    from_term: str
    to_term: str
    text: str
    chunk_ids: list[str]
    evidence_quote: str


class RelationOutput(BaseModel):
    relations: list[ExtractedRelation]


# ── 용어 판정 (AI-3) ──


class LLMTermJudgment(BaseModel):
    term_id: int
    status: TermStatus
    evidence_quote: str | None = None
    reason: str


class TermJudgeOutput(BaseModel):
    judgments: list[LLMTermJudgment]


# ── 학생 (AI-3, AI-4, AI-5, AI-7) ──


class HandRaiseOutput(BaseModel):
    raise_hand: bool
    reason: str | None = None


class QuestionMakeOutput(BaseModel):
    question: str


class NoteLine(BaseModel):
    text: str
    message_refs: list[int]
    evidence_quote: str


class NoteConcept(BaseModel):
    term_id: int             # 용어 목록의 term_id (0 = 어느 용어에도 맞지 않음)
    lines: list[NoteLine]


class NoteQuestionTerm(BaseModel):
    number: int              # [질문 목록]의 번호
    term_id: int


class NoteWonder(BaseModel):
    index: int               # [손 들었지만 묻지 못한 메모] 번호
    term_id: int
    text: str                # 메모를 페르소나 말투의 질문 한 문장으로


class NoteOutput(BaseModel):
    concepts: list[NoteConcept]           # 용어별 개념 (선생님 설명 메시지)
    questions: list[NoteQuestionTerm]     # 학생 질문마다 어느 용어에 대한 것인지
    wonders: list[NoteWonder]             # 확인하지 않은 질문 (손 들었지만 선생님이 받지 않은 것)


class LLMSolveAnswer(BaseModel):
    number: int
    knows: bool
    response: str | None = None
    message_refs: list[int]
    evidence_quote: str | None = None


class ExamSolveOutput(BaseModel):
    answers: list[LLMSolveAnswer]


# ── 평가 (AI-6, AI-8) ──


class LLMQuestion(BaseModel):
    slot: int
    question_type: QuestionType
    content: str
    choices: list[str] | None = None
    correct_answer: str
    explanation: str
    evidence_lines: list[int]
    evidence_quote: str


class ExamGenerationOutput(BaseModel):
    questions: list[LLMQuestion]


class AnswerCheck(BaseModel):
    # comment를 결론보다 먼저 둔다 (ClaimCheck와 같은 이유: 따져 본 뒤에 결론을 내게)
    slot: int
    comment: str
    answer_supported: bool
    single_correct_answer: bool


class AnswerVerificationOutput(BaseModel):
    checks: list[AnswerCheck]


class LLMGrade(BaseModel):
    number: int
    is_correct: bool
    reason: str


class GradingOutput(BaseModel):
    results: list[LLMGrade]


# ── 보고서 (AI-9) ──


class LLMStrength(BaseModel):
    text: str
    message_refs: list[int]
    quote: str | None = None
    question_numbers: list[int]


class LLMRecommendation(BaseModel):
    term_id: int
    text: str
    # 자료(정의)와 어긋나게 설명한 부분 / 같은 설명 중 맞게 한 부분 — 선생님 메시지 원문 인용 (코드가 확인)
    wrong_quote: str | None = None
    right_quote: str | None = None


class ReportOutput(BaseModel):
    strengths: list[LLMStrength]
    review_recommendations: list[LLMRecommendation]
