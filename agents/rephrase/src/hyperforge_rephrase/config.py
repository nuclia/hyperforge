from typing import Dict, List, Literal, Optional

from hyperforge.agent import AgentConfig
from hyperforge.llm_config import LLMConfig, LLMField, llm_defaults
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
    model: LLMField = Field(
        default=LLMConfig(model_id=llm_defaults.default),
        title="Generative model",
        description="Model used to generate the rephrased question",
    )
    module: Literal["rephrase"] = "rephrase"
    split_question: bool = False
    rules: Optional[List[str]] = None
