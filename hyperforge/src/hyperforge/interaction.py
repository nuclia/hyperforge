import uuid
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from hyperforge.models import (
    Answer,
    AnswerCitations,
    Context,
    Step,
    Visualization,
)

# Models used between client and API for itneraction requests/responses/streams


class Operation(int, Enum):
    START = 0
    STOP = 1


class AnswerOperation(int, Enum):
    ANSWER = 0
    START = 2
    DONE = 3
    ERROR = 4
    AGENT_REQUEST = 5
    ANSWER_CHUNK = 6
    REASONING = 7


class StreamingChunk(BaseModel):
    """A single streaming chunk of text produced by an LLM.

    When last=True this is the final chunk in the stream
    """

    text: str
    last: bool = False


class ARAGException(BaseModel):
    detail: str
    extra: Optional[Dict[str, Any]] = None


class ValidationFeedbackSchema(BaseModel):
    call_tool: bool


class PromptFeedbackSchema(BaseModel):
    prompt_id: str
    data: Any


class Provider(Enum):
    GOOGLE_OAUTH = "google_oauth"
    AZURE_OAUTH = "azure_oauth"
    AZURE_CERTIFICATE_CREDENTIALS = "azure_certificate_credentials"
    AWS_S3_ACCESS_KEYS = "aws_s3_access_keys"
    SHAREFILE_OAUTH = "sharefile_oauth"


class OAuthAuthenticateURL(BaseModel):
    oauth_url: str


class OAuthFeedbackReturnSchema(BaseModel):
    existing_credentials: Optional[Dict[str, Dict[str, str]]] = None


class Feedback(BaseModel):
    request_id: str
    feedback_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    question: str
    module: str
    agent_id: str
    data: Any
    timeout_ms: int = 10_000
    response_schema: Any
    get_credentials: Optional[Dict[str, Provider]] = None
    credentials: Optional[Dict[str, Dict[str, Any]]] = None


class AragAnswer(BaseModel):
    exception: Optional[ARAGException] = None
    answer: Optional[str] = None
    answer_citations: Optional[AnswerCitations] = None
    answer_urls: Optional[list[str]] = None
    agent_request: Optional[str] = None
    generated_text: Optional[str] = None
    step: Optional[Step] = None
    possible_answer: Optional[Answer] = None
    context: Optional[Context] = None
    operation: AnswerOperation = AnswerOperation.ANSWER
    seqid: Optional[int] = None
    original_question_uuid: Optional[str] = None
    actual_question_uuid: Optional[str] = None
    feedback: Optional[Feedback] = None
    oauth: Optional[OAuthAuthenticateURL] = None
    data_visualizations: Optional[list[Visualization]] = None
    streaming_response_chunk: Optional[StreamingChunk] = None
    reasoning: Optional[StreamingChunk] = None

    def __str__(self) -> str:
        if self.step is not None:
            return "\033[1mStep: \033[0m \n" + str(self.step)
        elif self.exception is not None:
            return "\033[1mException: \033[0m \n" + str(self.exception)
        elif self.context is not None:
            return "\033[1mContext: \033[0m \n" + str(self.context)
        return (
            f"AragAnswer(operation={self.operation}, answer={self.answer}, "
            f"agent_request={self.agent_request}, exception={self.exception})"
        )
