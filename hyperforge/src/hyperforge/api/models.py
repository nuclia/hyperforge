from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from nucliadb_models import TextFormat
from pydantic import BaseModel, Field

from hyperforge.driver import DriverConfig
from hyperforge.models import Rules


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
