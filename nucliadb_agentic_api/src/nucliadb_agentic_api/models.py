from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from hyperforge.models import MemoryConfig, Rules
from typing_extensions import Annotated

from hyperforge.driver import DriverConfig
from nucliadb_models import TextFormat
from pydantic import BaseModel, Field


class StashRoles(str, Enum):
    # Can do anything at the stash
    OWNER = "SOWNER"

    # Can access the stash
    MEMBER = "SMEMBER"

    # Can access the stash
    CONTRIBUTOR = "SCONTRIBUTOR"


class AccountRoles(str, Enum):
    OWNER = "AOWNER"
    MEMBER = "AMEMBER"


class AgentRole(str, Enum):
    MEMBER = "SESSIONMEMBER"


class NucliaDBRoles(str, Enum):
    MANAGER = "MANAGER"
    READER = "READER"
    WRITER = "WRITER"


class UserType(str, Enum):
    ROOT = "ROOT"
    DEALER = "DEALER"
    USER = "USER"
    READONLY = "READONLY"
    MANAGER = "MANAGER"
    SALES = "SALES"


class AccountTypes(str, Enum):
    TRIAL = "stash-trial"
    STARTER = "stash-starter"
    GROWTH = "stash-growth"
    STARTUP = "stash-startup"
    ENTERPRISE = "stash-enterprise"

    # will be removed at some point in the near future
    DEVELOPER = "stash-developer"
    BUSINESS = "stash-business"

    # V3 account types
    V3_STARTER = "v3starter"
    V3_FLY = "v3fly"
    V3_GROWTH = "v3growth"
    V3_PRO = "v3pro"
    V3_ENTERPRISE = "v3enterprise"
    COWORK = "cowork"


class SessionData(BaseModel):
    slug: str
    name: str
    summary: str
    data: str
    format: TextFormat


INFO_FIELD_ID = "info"

DEFAULT_RESOURCE_LIST_PAGE_SIZE = 20


class InspectData(BaseModel):
    contexts: List[Any]
    driver: List[DriverConfig]
    postprocess: List[Any]
    preprocess: List[Any]
    rules: Rules


class AgentID(BaseModel):
    id: str


class DriverID(BaseModel):
    id: str


class PromptID(BaseModel):
    id: str


class InteractionsAuditDownloadRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Filter by session ID")
    year: Optional[int] = Field(
        default=None,
        description="Filter by year (e.g., 2024). If not specified, defaults to the current year.",
    )
    month: Optional[int] = Field(
        default=None,
        description="Filter by month (1-12). If not specified, defaults to the past month.",
    )


class DownloadStatus(BaseModel):
    id: str
    type: str
    status: Literal["pending", "ready"]
    download_url: str | None
    query: dict[str, Any]


class InteractionOperation(int, Enum):
    QUESTION = 0
    QUIT = 1


class InteractionRequest(BaseModel):
    question: str
    headers: Dict[str, str] = {}
    arguments: Dict[str, str] = {}
    operation: InteractionOperation = InteractionOperation.QUESTION
    streaming: bool = False


class AgenticRephraseConfiguration(BaseModel):
    ask_to: Optional[str] = None
    prompt: Optional[str] = None
    model: Optional[str] = None


class AgenticSmartAgentMode(str, Enum):
    REACTIVE = "reactive"
    PLAN_EXECUTE = "plan_execute"


class AgenticSmartAgentModels(BaseModel):
    context_validation: Optional[str] = None
    planner: Optional[str] = None
    executor: Optional[str] = None


class NucliaDBAgenticSource(BaseModel):
    type: Literal["nucliadb"] = "nucliadb"
    description: Optional[str] = None
    filter_expression: Optional[str] = None
    connection: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


class GoogleAgenticSource(BaseModel):
    type: Literal["google"] = "google"
    description: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


class PerplexityAgenticSource(BaseModel):
    type: Literal["perplexity"] = "perplexity"
    description: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


class MCPAgenticSource(BaseModel):
    type: Literal["mcp"] = "mcp"
    description: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


class SyncAgenticSource(BaseModel):
    type: Literal["sync"] = "sync"
    description: Optional[str] = None
    connection: Optional[str] = None
    params: Optional[Dict[str, Any]] = None


AgenticSource = Annotated[
    NucliaDBAgenticSource
    | GoogleAgenticSource
    | PerplexityAgenticSource
    | MCPAgenticSource
    | SyncAgenticSource,
    Field(discriminator="type"),
]


class AgenticSmartAgentConfiguration(BaseModel):
    mode: AgenticSmartAgentMode = AgenticSmartAgentMode.REACTIVE
    extra_prompt: Optional[str] = None
    models: Optional[AgenticSmartAgentModels] = None
    sources: List[AgenticSource] = Field(default_factory=list)


class AgenticSummarizeConfiguration(BaseModel):
    user_prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    conversational: bool = False
    model: Optional[str] = None


class AgenticConfiguration(BaseModel):
    rephrase: Optional[AgenticRephraseConfiguration] = None
    smart_agent: Optional[AgenticSmartAgentConfiguration] = None
    summarize: Optional[AgenticSummarizeConfiguration] = None


class AgenticConfigSchema(BaseModel):
    title: Optional[str] = None
    config: Optional[AgenticConfiguration] = None
