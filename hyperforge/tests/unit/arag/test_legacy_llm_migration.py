import json
from uuid import uuid4

from sqlalchemy import text

from hyperforge.db.legacy_llm_migration import (
    migrate_legacy_llm_fields,
    wrap_legacy_llm_fields,
)


def test_wraps_nested_agent_fields_without_touching_other_models():
    config = {
        "module": "smart",
        "planner_model": "chatgpt-azure-4o/custom-id",
        "executor_model": {
            "_type": "llm_config",
            "model_id": "existing",
            "reasoning": "enabled",
        },
        "registered_agents": [
            {
                "module": "multi_mcp",
                "rephrase_model": "old-rephrase",
                "configs": [{"module": "mcp", "sampling_model": "old-sampling"}],
            },
            {"module": "data_viz", "model": "external-string"},
            {
                "module": "sql",
                "conversion_model": "external-string",
                "rephrase_model": "old-context",
            },
            {"module": "salesforce", "sampling_model": "old-plugin-inherited"},
            {
                "module": "context_conditional",
                "then": [{"module": "rephrase", "model": "old"}],
            },
        ],
        "metadata": {"module": "rephrase", "model": "not-a-nested-agent"},
    }

    assert wrap_legacy_llm_fields(config)
    assert config["planner_model"] == {
        "_type": "llm_config",
        "model_id": "chatgpt-azure-4o/custom-id",
    }
    assert config["executor_model"]["reasoning"] == "enabled"
    assert (
        config["registered_agents"][0]["rephrase_model"]["model_id"] == "old-rephrase"
    )
    assert (
        config["registered_agents"][0]["configs"][0]["sampling_model"]["model_id"]
        == "old-sampling"
    )
    assert config["registered_agents"][1]["model"] == "external-string"
    assert config["registered_agents"][2]["conversion_model"] == "external-string"
    assert config["registered_agents"][2]["rephrase_model"]["model_id"] == "old-context"
    assert (
        config["registered_agents"][3]["sampling_model"]["model_id"]
        == "old-plugin-inherited"
    )
    assert config["registered_agents"][4]["then"][0]["model"]["model_id"] == "old"
    assert config["metadata"]["model"] == "not-a-nested-agent"
    assert not wrap_legacy_llm_fields(config)


def test_migrate_legacy_fields_in_stored_configs(test_db):
    row_id = str(uuid4())
    test_db.execute(
        text(
            "INSERT INTO retrieval_agent_config (account, agent_id, rules, memory) "
            "VALUES (:account, :agent_id, '{}'::jsonb, '{}'::jsonb)"
        ),
        {"account": row_id, "agent_id": row_id},
    )
    test_db.execute(
        text(
            "INSERT INTO retrieval_agent_workflow (account, agent_id, workflow_id, name) "
            "VALUES (:account, :agent_id, 'default', 'default')"
        ),
        {"account": row_id, "agent_id": row_id},
    )
    test_db.execute(
        text(
            "INSERT INTO retrieval_agent_preprocess "
            "(id, account, agent_id, workflow_id, preprocess) "
            "VALUES (:id, :account, :agent_id, 'default', CAST(:config AS jsonb))"
        ),
        {
            "id": row_id,
            "account": row_id,
            "agent_id": row_id,
            "config": json.dumps(
                {"module": "rephrase", "model": "chatgpt-azure-4o-mini"}
            ),
        },
    )

    assert migrate_legacy_llm_fields(test_db) == 1
    assert migrate_legacy_llm_fields(test_db) == 0
    config = test_db.execute(
        text("SELECT preprocess FROM retrieval_agent_preprocess WHERE id = :id"),
        {"id": row_id},
    ).scalar_one()
    assert config["model"] == {
        "_type": "llm_config",
        "model_id": "chatgpt-azure-4o-mini",
    }
