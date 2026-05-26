from typing import ClassVar, Dict, List, Literal, Optional

from hyperforge.driver import DriverConfig, EncryptedPayload
from nucliadb_models.filters import CatalogFilterExpression, FilterExpression
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict


class ManagerConnection(BaseModel):
    api_key: Optional[str]
    url: str
    headers: Dict[str, str]


class NucliaDBConnection(EncryptedPayload):
    encrypted_fields: ClassVar[list[str]] = ["key"]
    key: Optional[str] = None
    url: str
    manager: str
    filters: List[str] = Field(
        default_factory=list,
        deprecated=True,
        description="Use filter_expression instead",
    )
    filter_expression: Optional[FilterExpression] = Field(
        default=None,
        description="Expression to filter fields/paragraphs when retrieving context. This will be combined with any other filters that are provided at query time or chosen by the agent, using an AND operator.",
    )
    catalog_filter_expression: Optional[CatalogFilterExpression] = Field(
        default=None,
        description="Expression to filter catalog items when retrieving context. This will be combined with any other filters that are provided at query time or chosen by the agent, using an AND operator.",
    )
    description: str = Field(
        description="Description of the knowledge box, used to give context to the agent and assist in selecting the right knowledge box for a given query."
    )
    kbid: str


class NucliaDBConfig(DriverConfig[NucliaDBConnection]):
    model_config = ConfigDict(title="Knowledge Box")
    provider: Literal["nucliadb"]
    config: NucliaDBConnection
