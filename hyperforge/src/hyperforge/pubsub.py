from typing import Annotated, Dict, Literal

from pydantic import AliasChoices, BaseModel, Field
from pydantic.types import Discriminator, Tag

from hyperforge.interaction import (
    AragAnswer,
    Feedback,
    OAuthAuthenticateURL,
)

# Messages used in the pubsub protocol between API and agent servers


class StartInteraction(BaseModel):
    """Starts an interaction for a given agent/session"""

    account: str
    agent_id: str = Field(validation_alias=AliasChoices("agent_id", "kbid"))
    session: str
    question_id: str
    question: str
    headers: Dict[str, str] = {}
    arguments: Dict[str, str] = {}
    workflow_id: str = "default"
    streaming: bool = False
    op: Literal["start"] = "start"


class UserToAgentInteraction(BaseModel):
    """Sends an user response"""

    op: Literal["user_response"] = "user_response"
    request_id: str
    response: str


class AgentPing(BaseModel):
    """Periodic ping to indicate the agent is still working"""

    op: Literal["ping"] = "ping"


class AgentDone(BaseModel):
    """Indicates the agent has finished and no more messages will be sent"""

    op: Literal["done"] = "done"


class AgentToUserRequest(BaseModel):
    """Sends a partial or final answer"""

    op: Literal["agent_request"] = "agent_request"
    feedback: Feedback


class OAuthRequest(BaseModel):
    """Sends a partial or final answer"""

    op: Literal["oauth"] = "oauth"
    oauth: OAuthAuthenticateURL


class AgentAnswer(BaseModel):
    """Sends a partial or final answer"""

    op: Literal["answer"] = "answer"
    answer: AragAnswer


# Messages send from the agent server to the API server, to be passed to the client
AgentMessage = Annotated[
    Annotated[AgentPing, Tag("ping")]
    | Annotated[AgentDone, Tag("done")]
    | Annotated[OAuthRequest, Tag("oauth")]
    | Annotated[AgentAnswer, Tag("answer")]
    | Annotated[AgentToUserRequest, Tag("agent_request")]
    | Annotated[UserToAgentInteraction, Tag("user_response")],
    Discriminator("op"),
]


class QuitRequest(BaseModel):
    """Requests the agent to stop (the client has abandoned the session)"""

    op: Literal["quit"] = "quit"


# Messages sent from the API to the agent server
APIMessage = Annotated[Annotated[QuitRequest, Tag("quit)")], Discriminator("op")]
