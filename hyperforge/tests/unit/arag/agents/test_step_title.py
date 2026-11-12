import pytest
from hyperforge_generate.config import GenerateAgentConfig
from hyperforge_mcp.config import MCPAgentConfig
from hyperforge_nucliadb.ask.config import AskAgentConfig
from hyperforge_remi.config import RemiAgentConfig
from hyperforge_rephrase.config import RephraseAgentConfig
from hyperforge_smart.config import SmartAgentConfig
from hyperforge_summarize.config import SummarizeAgentConfig

from hyperforge.agent import Agent, AgentConfig


class StubAgent(Agent):
    def __init__(self, config: AgentConfig):
        self.config = config


@pytest.mark.parametrize(
    "config, description, expected",
    [
        (MCPAgentConfig(source="my-mcp"), "Tool selection", "MCP: Tool selection"),
        (AskAgentConfig(), "Choose parameters", "Knowledge Box Ask: Choose parameters"),
        (SmartAgentConfig(), "Agent selection", "Smart agent: Agent selection"),
        (GenerateAgentConfig(), "Generate", "Generate: Generate"),
        (SummarizeAgentConfig(), "Summarize", "Summarize: Summarize"),
        (RephraseAgentConfig(), "Rephrase", "Rephrase: Rephrase"),
        (RemiAgentConfig(), "REMi evaluation", "REMi evaluation: REMi evaluation"),
    ],
)
def test_step_title_uses_schema_title(config, description, expected):
    assert StubAgent(config).step_title(description) == expected


def test_step_title_falls_back_to_instance_title():
    config = AgentConfig(module="stub", title="my-custom-bot")
    assert StubAgent(config).step_title("Do something") == "my-custom-bot: Do something"


def test_step_title_default_instance_title():
    assert (
        StubAgent(AgentConfig(module="stub")).step_title("Do something")
        == "agent: Do something"
    )


def test_step_title_schema_title_not_overridden_by_instance_title():
    config = MCPAgentConfig(source="my-mcp", title="user-custom-title")
    assert StubAgent(config).step_title("Tool selection") == "MCP: Tool selection"
