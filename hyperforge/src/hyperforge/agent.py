import abc
import uuid
from typing import Any, ClassVar, Dict, Generic, List, Optional, Self, TypeVar

from pydantic import BaseModel, Field

from hyperforge.definition import FunctionDefinition
from hyperforge.manager import Manager
from hyperforge.memory.memory import QuestionMemory
from hyperforge.utils import WidgetType


class AgentConfig(BaseModel):
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = "agent"
    rules: Optional[List[str]] = Field(
        default=None,
        title="Agent rules",
        description="List of rules to follow when executing this agent",
        json_schema_extra={
            "widget": WidgetType.NOT_SHOWN,
        },
    )
    max_retries: int = 1
    module: Any = Field(
        ..., title="Agent module", description="Module/type of the agent"
    )

    def subagent_configure(self, config: dict):
        pass


T_Config = TypeVar("T_Config", bound=AgentConfig)


class Agent(Generic[T_Config]):
    __root_agent__: bool = False
    __published_functions__: ClassVar[Dict[str, FunctionDefinition]] = {}
    agent_id: str
    config: T_Config

    def __init__(self, config: T_Config, agent_id: Optional[str] = None):
        self.config: T_Config = config
        self.agent_id: str = (
            agent_id
            if agent_id is not None
            else config.id
            if hasattr(config, "id") and config.id is not None
            else str(uuid.uuid4())
        )

    @abc.abstractmethod
    async def inner_from_config(self, config: T_Config, agent_id: Optional[str] = None):
        """Initialize the agent from the config. This is where you should put any
        async initialization code that needs to run when the agent is created.
        """
        pass

    @classmethod
    async def from_config(
        cls, config: T_Config, agent_id: Optional[str] = None
    ) -> Self:
        instance = cls(config=config, agent_id=agent_id)
        await instance.inner_from_config(config, agent_id)
        return instance

    def step_title(self, description: str) -> str:
        """Format a step title as '<agent title>: <description>'.

        Uses the user-configured instance title if set, otherwise falls back to
        the Pydantic model_config title (e.g. 'MCP', 'SQL query') which describes
        the agent type.
        """
        config = self.config  # type: ignore[attr-defined]
        title = type(config).model_config.get("title") or config.title  # type: ignore[attr-defined]
        return f"{title}: {description}"

    async def __call__(
        self,
        memory: QuestionMemory,
        manager: Manager,
    ):
        raise NotImplementedError()
