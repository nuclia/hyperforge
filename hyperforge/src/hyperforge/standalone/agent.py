"""
StaticAgentManager — a drop-in replacement for AgentManager that serves agent
configuration from an in-memory StandaloneConfig rather than a PostgreSQL database.

Only the methods actually called at runtime (by SessionManager and the MCP/interaction
endpoints) are implemented.  Management mutations (add/delete/patch) intentionally
raise NotImplementedError — the standalone deployment is read-only with respect to
agent configuration.
"""

import datetime
from typing import Any, List

from hyperforge_database import exceptions
from hyperforge_standalone.config import StandAloneAgentConfig

from hyperforge.memory import MemoryConfig, Rules
from hyperforge.prompts import PromptConfig
from hyperforge.retrieval.config import RetrievalAgentConfig
from hyperforge.workflows import RetrievalAgent, WorkflowData

_EPOCH = datetime.datetime.now()


class StaticAgentManager:
    _config: dict[str, StandAloneAgentConfig]

    def __init__(self, config: dict[str, StandAloneAgentConfig]) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Lifecycle (no-ops — nothing to connect/disconnect)
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        pass

    async def finalize(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_agent(self, agent_id: str) -> StandAloneAgentConfig:
        agent_config = self._config.get(agent_id)
        if not agent_config:
            raise exceptions.NotFoundError(f"Agent '{agent_id}' not found")
        return agent_config

    async def ensure_workflow_active(
        self, account: str, agent_id: str, workflow_id: str
    ) -> None:
        agent_config = self._get_agent(agent_id)
        if workflow_id not in agent_config.workflows:
            raise exceptions.NotFoundError("Workflow not found")

    def _build_retrieval_config(
        self, agent_config: StandAloneAgentConfig, workflow_id: str = "default"
    ) -> RetrievalAgentConfig:
        """Merge StandAloneAgentConfig + WorkflowConfig into a RetrievalAgentConfig."""
        workflow = agent_config.workflows.get(workflow_id)
        if workflow is None:
            raise exceptions.NotFoundError(
                f"Workflow '{workflow_id}' not found in agent"
            )
        workflow_data = workflow.as_workflow_data(workflow_id)
        # Merge agent-level rules with workflow-level rules (workflow takes precedence).
        merged_rules = workflow.rules if workflow.rules.rules else agent_config.rules
        return RetrievalAgentConfig(
            drivers=agent_config.drivers,
            rules=merged_rules,
            memory=MemoryConfig(),
            workflow=workflow_data,
            preprocess=workflow.preprocess,
            context=workflow.context,
            generation=workflow.generation,
            postprocess=workflow.postprocess,
        )

    # ------------------------------------------------------------------
    # Methods used by SessionManager (session.py)
    # ------------------------------------------------------------------

    async def get_driver(self, account: str, agent_id: str, driver: str) -> Any:
        agent_config = self._get_agent(agent_id)
        for drv in agent_config.drivers:
            if drv.identifier == driver:
                return drv
        raise exceptions.NotFoundError(
            f"Driver '{driver}' not found in agent '{agent_id}'"
        )

    async def get_drivers(self, account: str, agent_id: str) -> List[Any]:
        agent_config = self._get_agent(agent_id)
        return list(agent_config.drivers)

    async def workflows_list(self, account: str, agent_id: str) -> List[WorkflowData]:
        agent_config = self._get_agent(agent_id)
        return [
            workflow.as_workflow_data(workflow_id)
            for workflow_id, workflow in agent_config.workflows.items()
        ]

    async def get_agent_config_basic(
        self, account: str, agent_id: str
    ) -> RetrievalAgent:
        agent_config = self._get_agent(agent_id)
        # Use the default workflow for description/title fallbacks.
        default_workflow = agent_config.workflows.get("default")
        description = agent_config.description or (
            default_workflow.description if default_workflow else None
        )
        title = agent_config.title or agent_id
        return RetrievalAgent(
            account=account,
            agent_id=agent_id,
            memory=None,
            description=description,
            title=title,
            instructions=agent_config.instructions,
            created=_EPOCH,
            modified=_EPOCH,
        )

    async def get_agent_config(
        self,
        account: str,
        agent_id: str,
        internal_nucliadb_url: str | None = None,
        default_memory: bool = False,
        workflow_id: str = "default",
    ) -> RetrievalAgentConfig:
        agent_config = self._get_agent(agent_id)
        return self._build_retrieval_config(agent_config, workflow_id)

    async def get_prompt(
        self, agent_id: str, account: str, prompt_id: str
    ) -> PromptConfig:
        agent_config = self._get_agent(agent_id)
        for prompt in agent_config.prompts:
            if prompt.prompt_id == prompt_id:
                return prompt
        raise exceptions.NotFoundError(
            f"Prompt '{prompt_id}' not found in agent '{agent_id}'"
        )

    async def get_prompts(self, agent_id: str, account: str) -> List[PromptConfig]:
        agent_config = self._get_agent(agent_id)
        return list(agent_config.prompts)

    async def get_rules(self, account: str, agent_id: str) -> Rules:
        agent_config = self._get_agent(agent_id)
        return agent_config.rules

    async def get_preprocess(
        self, account: str, agent_id: str, workflow_id: str = "default"
    ) -> list:
        agent_config = self._get_agent(agent_id)
        workflow = agent_config.workflows.get(workflow_id)
        if workflow is None:
            return []
        return workflow.preprocess or []

    async def get_context(
        self, account: str, agent_id: str, workflow_id: str = "default"
    ) -> list:
        agent_config = self._get_agent(agent_id)
        workflow = agent_config.workflows.get(workflow_id)
        if workflow is None:
            return []
        return workflow.context or []

    async def get_generation(
        self, account: str, agent_id: str, workflow_id: str = "default"
    ) -> list:
        agent_config = self._get_agent(agent_id)
        workflow = agent_config.workflows.get(workflow_id)
        if workflow is None:
            return []
        return workflow.generation or []

    async def get_postprocess(
        self, account: str, agent_id: str, workflow_id: str = "default"
    ) -> list:
        agent_config = self._get_agent(agent_id)
        workflow = agent_config.workflows.get(workflow_id)
        if workflow is None:
            return []
        return workflow.postprocess or []
