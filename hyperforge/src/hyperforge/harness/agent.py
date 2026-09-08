from typing import Any, cast

from hyperforge.agent import Agent
from hyperforge.configure import get_agent_klass
from hyperforge.harness.config import HarnessAgentConfig
from hyperforge.harness_sdk import (
    AgentHarness,
    HarnessEventType,
    HarnessMessage,
    NucliaChatCompletionsClient,
    NucliaModelClient,
)
from hyperforge.interaction import Feedback, StreamingChunk
from hyperforge.llm import AsyncNuaClient
from hyperforge.manager import Manager
from hyperforge.memory.memory import QuestionMemory


class HarnessAgent:
    def __init__(self, config: HarnessAgentConfig, agents: list[Agent[Any]]) -> None:
        self.config = config
        self.agents = agents

    @classmethod
    async def from_config_class(cls, config: HarnessAgentConfig) -> "HarnessAgent":
        agents = [
            await get_agent_klass(agent_config.module).from_config(agent_config)
            for agent_config in config.agents
        ]
        return cls(config, agents)

    async def __call__(self, memory: QuestionMemory, manager: Manager) -> None:
        for agent in self.agents:
            preload = getattr(agent, "preload", None)
            if callable(preload):
                await preload(manager, memory)
        tools = [
            tool
            for agent in self.agents
            for tool in AgentHarness.to_tools(
                agent, namespace=agent.agent_id, manager=manager
            )
        ]
        rules = [
            rule if isinstance(rule, str) else rule.prompt
            for rule in memory.get_rules()
            if isinstance(rule, str) or rule.prompt
        ]
        system_prompt = self.config.system_prompt
        if rules:
            system_prompt += "\n\nRules:\n" + "\n".join(f"- {rule}" for rule in rules)
        harness = AgentHarness(
            model=self.config.model,
            model_client=NucliaModelClient(
                NucliaChatCompletionsClient(cast(AsyncNuaClient, manager.nua))
            ),
            reasoning_effort=self.config.reasoning_effort,
            tools=tools,
            system_prompt=system_prompt,
            conversation_id=memory.get_session_id(),
            disabled_core_tools=[
                *self.config.disabled_core_tools,
                # Since we're not using the storage adapter implementation,
                # we need to disable these tools that require storage access
                # across sessions.
                "remember",
                "recall",
                "forget",
            ],
            execution_context={
                "user_id": memory.session.user_info.get("id", "system"),
                "conversation_metadata": {
                    "agent_id": memory.get_agent_id(),
                    "workflow_id": memory.get_workflow_id(),
                    "question_id": memory.original_question_uuid,
                },
                "memory": memory,
            },
            feedback_enabled=self.config.feedback_enabled,
            usage_limits=self.config.usage_limits,
        )
        history = await memory.get_chat_history()
        harness.add_messages(
            HarnessMessage(
                role="user" if message.author.value == "USER" else "assistant",
                content=message.text,
            )
            for message in history
        )
        result = ""
        async for event in harness.run(memory.original_question):
            if event.agent_id != harness.agent_id:
                continue
            if event.type == HarnessEventType.TEXT_DELTA:
                await memory.emit_streaming_chunk(
                    StreamingChunk(text=event.payload["text"])
                )
            elif event.type == HarnessEventType.REASONING_DELTA:
                await memory.emit_streaming_chunk(
                    StreamingChunk(text=event.payload["text"]), reasoning=True
                )
            elif event.type == HarnessEventType.FEEDBACK_REQUESTED:
                response = await memory.send_feedback(
                    Feedback(
                        request_id=event.payload["request_id"],
                        question=event.payload["question"],
                        module="harness",
                        agent_id=memory.get_agent_id(),
                        data={},
                        timeout_ms=event.payload["timeout_ms"],
                        response_schema=event.payload["response_schema"],
                    )
                )
                if response is not None:
                    await harness.respond_feedback(
                        event.payload["request_id"], response.response
                    )
            elif event.type == HarnessEventType.TURN_COMPLETED:
                result = str(event.payload.get("text", ""))
        await memory.emit_streaming_chunk(StreamingChunk(text="", last=True))
        await memory.add_answer(result, "harness", "/harness")
        await memory.add_final_answer()
