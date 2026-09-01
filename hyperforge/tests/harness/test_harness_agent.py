from collections.abc import AsyncIterator
from typing import ClassVar

import pytest

from hyperforge.agent import Agent, AgentConfig
from hyperforge.configure import GLOBAL_REGISTRY, AgentRegistry
from hyperforge.definition import FunctionDefinition
from hyperforge.harness import HarnessAgent, HarnessAgentConfig
from hyperforge.manager import Manager
from hyperforge.memory.memory import BaseSessionMemory
from hyperforge.models import MemoryConfig, Rules


class PublishedConfig(AgentConfig):
    module: str = "published"


class PublishedAgent(Agent[PublishedConfig]):
    __published_functions__: ClassVar[dict[str, FunctionDefinition]] = {
        "lookup": FunctionDefinition(
            name="lookup",
            description="Look up a value.",
            parameters={"question": {"type": "string"}},
        )
    }

    async def inner_from_config(self, config, agent_id=None):
        pass

    async def lookup(self, question: str, memory=None) -> str:
        suffix = memory.headers.get("request-header", "")
        return f"{question.upper()}{suffix}"


class FakeNua:
    def __init__(self) -> None:
        self.calls = 0

    async def chat_completions_stream(self, payload, **kwargs) -> AsyncIterator[dict]:
        self.calls += 1
        assert any(
            message["role"] == "system" and "- Be concise" in message["content"]
            for message in payload["messages"]
        )
        assert not {"remember", "recall", "forget"} & {
            tool["function"]["name"] for tool in payload["tools"]
        }
        if self.calls == 1:
            tool = next(
                item
                for item in payload["tools"]
                if item["function"]["name"].endswith("__lookup")
            )
            yield {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "function": {
                                        "name": tool["function"]["name"],
                                        "arguments": '{"question":"hello"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        else:
            assert any(
                message["role"] == "tool" and "HELLO" in message["content"]
                for message in payload["messages"]
            )
            yield {"choices": [{"index": 0, "delta": {"content": "Final answer"}}]}


@pytest.mark.asyncio
async def test_harness_agent_runs_configured_legacy_agent_as_tool(monkeypatch) -> None:
    monkeypatch.setitem(
        GLOBAL_REGISTRY.agents,
        "published",
        AgentRegistry(
            id="published",
            agent_type="context",
            title="Published",
            description="Published test agent",
            config_schema=PublishedConfig,
            klass=PublishedAgent,
        ),
    )
    nua = FakeNua()
    config = HarnessAgentConfig.model_validate(
        {
            "model": "test-model",
            "agents": [{"id": "source", "module": "published"}],
            "workflow": {
                "id": "default",
                "name": "Default",
                "description": None,
                "parameters": None,
            },
        }
    )
    agent = await HarnessAgent.from_config_class(config)
    manager = Manager()
    manager.nua = nua
    session = BaseSessionMemory.from_config(
        MemoryConfig(),
        agent_id="test",
        workflow_id="default",
        rules=Rules(rules=["Be concise"]),
    )
    session.init("session")
    memory = session.start_question("Answer me", headers={"request-header": "!"})

    await agent(memory, manager)

    assert memory.final_answer == "Final answer"
    assert nua.calls == 2
