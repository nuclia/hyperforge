from typing import Any, cast

from hyperforge.agent import Agent
from hyperforge.configure import (
    create_toolkit_instance,
    get_agent_klass,
    get_registered_skills,
)
from hyperforge.harness.bridge import HarnessEventBridge
from hyperforge.harness.config import HarnessAgentConfig
from hyperforge.harness_sdk import (
    AgentHarness,
    BaseToolkit,
    HarnessMessage,
    HarnessSkill,
    HarnessStorageProtocol,
    NucliaChatCompletionsClient,
    NucliaModelClient,
    ToolkitRuntime,
    assemble_toolkits,
    codemode,
    create_codemode_tool,
)
from hyperforge.llm import AsyncNuaClient
from hyperforge.manager import Manager
from hyperforge.memory.memory import QuestionMemory
from hyperforge.prompts import prompt_config_to_skill, render_prompt_config


class HarnessAgent:
    def __init__(
        self,
        config: HarnessAgentConfig,
        agents: list[Agent[Any]],
        toolkits: list[BaseToolkit[Any]] | None = None,
        *,
        storage: HarnessStorageProtocol | None = None,
    ) -> None:
        self.config = config
        self.agents = agents
        self.toolkits = toolkits or []
        self.storage = storage

    @classmethod
    async def from_config_class(
        cls,
        config: HarnessAgentConfig,
        *,
        storage: HarnessStorageProtocol | None = None,
    ) -> "HarnessAgent":
        agents = [
            await get_agent_klass(agent_config.module).from_config(agent_config)
            for agent_config in config.agents
        ]
        toolkits = [
            await create_toolkit_instance(toolkit_config)
            for toolkit_config in config.toolkits
        ]
        return cls(config, agents, toolkits, storage=storage)

    async def __call__(self, memory: QuestionMemory, manager: Manager) -> None:
        for agent in self.agents:
            preload = getattr(agent, "preload", None)
            if callable(preload):
                await preload(manager, memory)

        model_client = NucliaModelClient(
            NucliaChatCompletionsClient(cast(AsyncNuaClient, manager.nua))
        )
        history = await memory.get_chat_history()
        rules_list = [
            rule if isinstance(rule, str) else rule.prompt
            for rule in memory.get_rules()
            if isinstance(rule, str) or rule.prompt
        ]
        workflow_context = {
            "question": memory.original_question,
            "arguments": dict(memory.arguments),
            "history": [
                {
                    "role": "user" if msg.author.value == "USER" else "assistant",
                    "content": msg.text,
                }
                for msg in history
            ],
            "rules": [rule for rule in rules_list if rule is not None],
            "user_info": dict(memory.session.user_info),
        }
        execution_context: dict[str, Any] = {
            "user_id": memory.session.user_info.get("id", "system"),
            "conversation_metadata": {
                "agent_id": memory.get_agent_id(),
                "workflow_id": memory.get_workflow_id(),
                "question_id": memory.original_question_uuid,
            },
            "memory": memory,
            "manager": manager,
            "drivers": manager.drivers,
            "nua_client": manager.nua,
            "workflow_context": workflow_context,
        }
        runtime = ToolkitRuntime(
            drivers=manager.drivers,
            model_client=model_client,
            nua_client=manager.nua,
            execution_context=execution_context,
        )
        toolkit_tools, toolkit_skills = await assemble_toolkits(
            self.toolkits, runtime=runtime
        )

        legacy_tools = [
            item
            for agent in self.agents
            for item in AgentHarness.to_tools(
                agent, namespace=agent.agent_id, manager=manager
            )
        ]
        tools = [*legacy_tools, *toolkit_tools]
        configured_codemode = (
            create_codemode_tool(
                limits=self.config.codemode_limits,
                remote_required=None,
            )
            if self.config.codemode_limits is not None
            else codemode
        )
        if self.config.codemode_enabled or self.config.execution_mode == "code":
            tools.append(configured_codemode)

        skills = self._collect_skills(toolkit_skills, memory)
        system_prompt = self._build_system_prompt(memory)

        disabled_core_tools = list(self.config.disabled_core_tools)
        if self.storage is None:
            disabled_core_tools.extend(["remember", "recall", "forget"])

        harness = AgentHarness(
            model=self.config.model or "code-workflow",
            model_client=model_client,
            reasoning_effort=self.config.reasoning_effort,
            tools=tools,
            skills=skills,
            system_prompt=system_prompt,
            conversation_id=memory.get_session_id(),
            disabled_core_tools=disabled_core_tools,
            storage=self.storage,
            execution_context=execution_context,
            feedback_enabled=self.config.feedback_enabled,
            usage_limits=self.config.usage_limits,
            procedural_guidance=self.config.procedural_guidance,
        )
        if self.storage is not None:
            await harness.load(create=True)
        if not harness.messages:
            harness.add_messages(
                HarnessMessage(
                    role="user" if message.author.value == "USER" else "assistant",
                    content=message.text,
                )
                for message in history
            )

        bridge = HarnessEventBridge(harness, memory)
        async with harness:
            if self.config.execution_mode == "code":
                assert self.config.workflow_code is not None
                event_stream = harness.run_code_workflow(
                    self.config.workflow_code,
                    memory.original_question,
                    codemode_tool=configured_codemode,
                    arguments=memory.arguments,
                )
            else:
                event_stream = harness.run(memory.original_question)
            async for event in event_stream:
                await bridge.handle_event(event)
        await bridge.finalize()

    def _collect_skills(
        self, toolkit_skills: list[HarnessSkill], memory: QuestionMemory
    ) -> list[HarnessSkill]:
        by_id: dict[str, HarnessSkill] = {}
        for registered in get_registered_skills():
            by_id[registered.id] = registered
        for toolkit_skill in toolkit_skills:
            by_id[toolkit_skill.id] = toolkit_skill
        if self.config.expose_prompts_as_skills:
            for prompt in self.config.prompts:
                if prompt.prompt_id and prompt.prompt_id == self.config.prompt_id:
                    continue
                if prompt.name == self.config.prompt_id:
                    continue
                converted = prompt_config_to_skill(
                    prompt,
                    arguments=memory.arguments or None,
                    preload=False,
                )
                by_id[converted.id] = converted
        for explicit in self.config.skills:
            by_id[explicit.id] = explicit
        return list(by_id.values())

    def _build_system_prompt(self, memory: QuestionMemory) -> str:
        system_prompt = self.config.system_prompt
        if self.config.prompt_id is not None:
            selected = next(
                (
                    item
                    for item in self.config.prompts
                    if item.prompt_id == self.config.prompt_id
                    or item.name == self.config.prompt_id
                ),
                None,
            )
            if selected is None:
                raise ValueError(f"Unknown prompt_id: {self.config.prompt_id}")
            rendered = render_prompt_config(selected, memory.arguments)
            system_prompt = f"{system_prompt}\n\n{rendered}"

        rules = [
            rule if isinstance(rule, str) else rule.prompt
            for rule in memory.get_rules()
            if isinstance(rule, str) or rule.prompt
        ]
        if rules:
            system_prompt += "\n\nRules:\n" + "\n".join(f"- {rule}" for rule in rules)
        return system_prompt
