from typing import Dict, List, Literal, Optional

from hyperforge.agent import AgentConfig
from hyperforge.utils import WidgetType
from pydantic import Field
from pydantic.config import ConfigDict


class RephraseAgentConfig(AgentConfig):
    model_config = ConfigDict(title="Rephrase")
    kb: Optional[str] = None
    rids: List[str] = Field(default_factory=list)
    labels: List[str] = Field(default_factory=list)
    synonyms: bool = True
    provided_synonyms: Dict[str, List[str]] = Field(
        default_factory=dict,
    )
    extend: bool = True
    session_info: bool = False
    history: bool = False
    model: str = Field(
        default="gemini-2.5-flash-lite",
        title="Generative model",
        description="Model used to generate the rephrased question",
        json_schema_extra={"widget": WidgetType.MODEL_SELECT},
    )
    module: Literal["rephrase"] = "rephrase"
    split_question: bool = False
    rules: Optional[List[str]] = None
