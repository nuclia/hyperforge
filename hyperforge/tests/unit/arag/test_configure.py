from unittest.mock import Mock, patch

from hyperforge.configure import (
    AgentRegistry,
    create_agent_schema,
    enabled_agent,
    enabled_driver,
)
from hyperforge.definition import FunctionDefinition


def test_create_agent_schema_includes_function_descriptions():
    agent_class = Mock(
        __published_functions__={
            "search": FunctionDefinition(
                name="search",
                description="Search for relevant documents",
                parameters={},
            )
        }
    )
    registry = AgentRegistry(
        id="test_agent",
        agent_type="context",
        title="Test agent",
        description="Agent used in tests",
        config_schema=Mock(model_json_schema=Mock(return_value={})),
        klass=agent_class,
    )

    schema = create_agent_schema(registry)

    assert schema["functions"] == {"search": "Search for relevant documents"}


def test_create_agent_schema_has_empty_functions_without_agent_class():
    registry = AgentRegistry(
        id="test_agent",
        agent_type="context",
        title="Test agent",
        description="Agent used in tests",
        config_schema=Mock(model_json_schema=Mock(return_value={})),
    )

    schema = create_agent_schema(registry)

    assert schema["functions"] == {}


def test_enabled_agent():
    agent = Mock(id="test_agent")
    with patch("hyperforge.configure.has_feature") as mock_has_feature:
        for env_check, account_check, enabled in (
            (True, True, True),
            (True, False, False),
            (False, True, True),
            (False, False, True),
        ):
            mock_has_feature.side_effect = [env_check, account_check]
            assert enabled_agent(agent, "env", "account_md5") == enabled


def test_enabled_driver():
    driver = Mock(id="test_driver")
    with patch("hyperforge.configure.has_feature") as mock_has_feature:
        for env_check, account_check, enabled in (
            (True, True, True),
            (True, False, False),
            (False, True, True),
            (False, False, True),
        ):
            mock_has_feature.side_effect = [env_check, account_check]
            assert enabled_driver(driver, "env", "account_md5") == enabled
