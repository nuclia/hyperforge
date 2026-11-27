"""
StaticAgentManager — serves unified `HarnessAgentConfig` workflows from an
in-memory `StandaloneConfig` rather than PostgreSQL.
"""

import datetime
import json
from typing import Any, List

from cryptography.fernet import Fernet, InvalidToken

from hyperforge.configure import GLOBAL_REGISTRY
from hyperforge.db import exceptions
from hyperforge.db.workflow_migration import transpile_stages_to_harness
from hyperforge.harness.config import HarnessAgentConfig
from hyperforge.harness_sdk import SYSTEM_PROMPT, HarnessSkill, ToolkitConfig
from hyperforge.llm_config import llm_defaults
from hyperforge.models import MemoryConfig, Rules
from hyperforge.prompts import PromptConfig
from hyperforge.standalone.config import StandAloneAgentConfig
from hyperforge.workflows import RetrievalAgent, WorkflowData

_EPOCH = datetime.datetime.now()


class StaticAgentManager:
    _config: dict[str, StandAloneAgentConfig]

    def __init__(self, config: dict[str, StandAloneAgentConfig]) -> None:
        self._config = config
        self._oauth_fernet = Fernet(Fernet.generate_key())
        self._oauth_credentials: dict[tuple[str, str, str, str, str], str] = {}

    async def initialize(self) -> None:
        pass

    async def finalize(self) -> None:
        pass

    def _get_agent(self, agent_id: str) -> StandAloneAgentConfig:
        agent_config = self._config.get(agent_id)
        if not agent_config:
            raise exceptions.NotFoundError(f"Agent '{agent_id}' not found")
        return agent_config

    async def get_sync_oauth_credentials(
        self,
        *,
        account: str,
        user_id: str,
        agent_id: str,
        provider: str,
        sync_config_id: str,
    ) -> dict[str, str] | None:
        key = (account, user_id, agent_id, provider, sync_config_id)
        encrypted_credentials = self._oauth_credentials.get(key)
        if encrypted_credentials is None:
            return None
        try:
            credentials = json.loads(
                self._oauth_fernet.decrypt(encrypted_credentials.encode()).decode()
            )
        except (InvalidToken, json.JSONDecodeError) as exc:
            raise ValueError("Invalid stored OAuth credentials") from exc
        if not isinstance(credentials, dict) or not all(
            isinstance(name, str) and isinstance(value, str)
            for name, value in credentials.items()
        ):
            raise ValueError("Invalid stored OAuth credentials")
        return credentials

    async def upsert_sync_oauth_credentials(
        self,
        *,
        account: str,
        user_id: str,
        agent_id: str,
        provider: str,
        sync_config_id: str,
        credentials: dict[str, str],
    ) -> None:
        key = (account, user_id, agent_id, provider, sync_config_id)
        payload = json.dumps(credentials, sort_keys=True, separators=(",", ":"))
        self._oauth_credentials[key] = self._oauth_fernet.encrypt(
            payload.encode()
        ).decode()

    async def upsert_sync_oauth_credentials_batch(
        self,
        *,
        account: str,
        user_id: str,
        agent_id: str,
        credentials_by_config: dict[str, tuple[str, dict[str, str]]],
    ) -> None:
        encrypted_credentials = {}
        for sync_config_id, (provider, credentials) in credentials_by_config.items():
            key = (account, user_id, agent_id, provider, sync_config_id)
            payload = json.dumps(credentials, sort_keys=True, separators=(",", ":"))
            encrypted_credentials[key] = self._oauth_fernet.encrypt(
                payload.encode()
            ).decode()
        self._oauth_credentials.update(encrypted_credentials)

    async def ensure_workflow_active(
        self, account: str, agent_id: str, workflow_id: str
    ) -> None:
        agent_config = self._get_agent(agent_id)
        if workflow_id not in agent_config.workflows:
            raise exceptions.NotFoundError("Workflow not found")

    def _build_harness_config(
        self, agent_config: StandAloneAgentConfig, workflow_id: str = "default"
    ) -> HarnessAgentConfig:
        workflow = agent_config.workflows.get(workflow_id)
        if workflow is None:
            raise exceptions.NotFoundError(
                f"Workflow '{workflow_id}' not found in agent"
            )
        merged_rules = workflow.rules if workflow.rules.rules else agent_config.rules
        combined_skills = [*agent_config.skills, *workflow.skills]

        has_native_harness = (
            workflow.execution_mode is not None
            or bool(workflow.toolkits)
            or bool(workflow.workflow_code)
            or workflow.model is not None
        )
        has_legacy_stages = any(
            (
                workflow.preprocess,
                workflow.context,
                workflow.generation,
                workflow.postprocess,
            )
        )

        if has_legacy_stages and not has_native_harness:
            ext_migrations = {
                k: v.handler
                for k, v in GLOBAL_REGISTRY.migrations.items()
                if v.handler is not None
            }
            compiled = transpile_stages_to_harness(
                preprocess=[x.model_dump() for x in workflow.preprocess],
                context=[x.model_dump() for x in workflow.context],
                generation=[x.model_dump() for x in workflow.generation],
                postprocess=[x.model_dump() for x in workflow.postprocess],
                external_migrations=ext_migrations,
            )
            harness_settings = dict(compiled["harness_config"])
            enabled_skill_ids = harness_settings.pop("enabled_skills", []) or []
            for sid in enabled_skill_ids:
                reg = GLOBAL_REGISTRY.skills.get(sid)
                if reg and reg.skill is not None:
                    combined_skills.append(reg.skill)
            return HarnessAgentConfig.model_validate(
                {
                    "execution_mode": compiled["execution_mode"],
                    "workflow_code": compiled["workflow_code"],
                    "toolkits": compiled["toolkits"],
                    "drivers": agent_config.drivers,
                    "skills": combined_skills,
                    "prompts": agent_config.prompts,
                    "rules": merged_rules,
                    "memory": MemoryConfig(),
                    **harness_settings,
                }
            )

        exec_mode = workflow.execution_mode or "loop"
        model_id = workflow.model or (llm_defaults.smart if exec_mode == "loop" else "")
        return HarnessAgentConfig(
            execution_mode=exec_mode,
            model=model_id,
            workflow_code=workflow.workflow_code,
            toolkits=workflow.toolkits,
            drivers=agent_config.drivers,
            skills=combined_skills,
            prompts=agent_config.prompts,
            system_prompt=(
                workflow.system_prompt or agent_config.instructions or SYSTEM_PROMPT
            ),
            feedback_enabled=workflow.feedback_enabled,
            codemode_enabled=workflow.codemode_enabled,
            usage_limits=workflow.usage_limits,
            procedural_guidance=workflow.procedural_guidance,
            rules=merged_rules,
            memory=MemoryConfig(),
        )

    async def get_driver(self, account: str, agent_id: str, driver: str) -> Any:
        agent_config = self._get_agent(agent_id)
        for drv in agent_config.drivers:
            if drv.identifier == driver:
                return drv
        raise exceptions.NotFoundError(
            f"Source '{driver}' not found in agent '{agent_id}'"
        )

    async def get_drivers(self, account: str, agent_id: str) -> List[Any]:
        agent_config = self._get_agent(agent_id)
        return list(agent_config.drivers)

    async def get_toolkits(
        self, account: str, agent_id: str, workflow_id: str = "default"
    ) -> List[ToolkitConfig]:
        cfg = self._build_harness_config(self._get_agent(agent_id), workflow_id)
        return list(cfg.toolkits)

    async def get_skills(self, account: str, agent_id: str) -> List[HarnessSkill]:
        return list(self._get_agent(agent_id).skills)

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
    ) -> HarnessAgentConfig:
        agent_config = self._get_agent(agent_id)
        return self._build_harness_config(agent_config, workflow_id)

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
        return (workflow.preprocess or []) if workflow else []

    async def get_context(
        self, account: str, agent_id: str, workflow_id: str = "default"
    ) -> list:
        agent_config = self._get_agent(agent_id)
        workflow = agent_config.workflows.get(workflow_id)
        return (workflow.context or []) if workflow else []

    async def get_generation(
        self, account: str, agent_id: str, workflow_id: str = "default"
    ) -> list:
        agent_config = self._get_agent(agent_id)
        workflow = agent_config.workflows.get(workflow_id)
        return (workflow.generation or []) if workflow else []

    async def get_postprocess(
        self, account: str, agent_id: str, workflow_id: str = "default"
    ) -> list:
        agent_config = self._get_agent(agent_id)
        workflow = agent_config.workflows.get(workflow_id)
        return (workflow.postprocess or []) if workflow else []
