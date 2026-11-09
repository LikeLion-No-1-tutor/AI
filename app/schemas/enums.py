"""모든 상태값은 이 Enum 문자열로만 주고받는다 (API 명세·ERD와 같은 값).

LLM 출력도 같은 Enum으로 검증하므로 목록에 없는 값이 오면 서버에서 거부하고 다시 부른다.
"""
from enum import Enum


class Sender(str, Enum):          # messages.sender
    TEACHER = "teacher"
    STUDENT = "student"


class MessageKind(str, Enum):     # messages.kind
    GREETING = "greeting"         # 학생 첫 인사
    EXPLANATION = "explanation"   # 사용자 설명
    QUESTION = "question"         # 손 든 표시를 눌러 만든 학생 질문
    ANSWER = "answer"             # 질문에 대한 사용자 답변


class TermStatus(str, Enum):      # lesson_terms.status (올라가기만 함)
    TODO = "todo"
    MENTIONED = "mentioned"
    EXPLAINED = "explained"


TERM_RANK = {TermStatus.TODO: 0, TermStatus.MENTIONED: 1, TermStatus.EXPLAINED: 2}


class ExamKind(str, Enum):        # exams.kind
    EVALUATION = "evaluation"     # 학생 AI가 풂
    RETEST = "retest"             # 사용자가 풂


class QuestionType(str, Enum):    # exam_questions.question_type
    MULTIPLE_CHOICE = "multiple_choice"   # 보기 4개, 정답은 보기 인덱스(0부터) 문자열
    SHORT_ANSWER = "short_answer"         # 주관식, 정답은 모범 답안


class SourceKind(str, Enum):      # AI 서버 내부: 자료 종류
    PDF = "pdf"
    TEXT = "text"
    CODE = "code"
    IMAGE = "image"
