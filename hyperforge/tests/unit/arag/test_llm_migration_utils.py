"""Tests for hyperforge.db.llm_migration_utils."""

import json
import uuid

import pytest
from hyperforge.db.llm_migration_utils import migrate_llm_models, walk_and_replace
from hyperforge.llm_config import LLM_CONFIG_TYPE
from sqlalchemy import text


class TestWalkAndReplace:
    def test_replaces_matching_model_id(self):
        config = {
            "planner_model": {"_type": LLM_CONFIG_TYPE, "model_id": "chatgpt-4.1"},
            "executor_model": {
                "_type": LLM_CONFIG_TYPE,
                "model_id": "gemini-2.5-flash",
            },
            "unrelated": "hello",
        }
        modified = walk_and_replace(
            config,
            {"chatgpt-4.1": "chatgpt-4.5", "gemini-2.5-flash": "gemini-3.0-flash"},
        )
        assert modified is True
        assert config["planner_model"]["model_id"] == "chatgpt-4.5"
        assert config["executor_model"]["model_id"] == "gemini-3.0-flash"
        assert config["unrelated"] == "hello"

    def test_preserves_uuid_suffix(self):
        custom_uuid = str(uuid.uuid4())
        config = {"_type": LLM_CONFIG_TYPE, "model_id": f"chatgpt-4.1/{custom_uuid}"}
        walk_and_replace(config, {"chatgpt-4.1": "chatgpt-4.5"})
        assert config["model_id"] == f"chatgpt-4.5/{custom_uuid}"

    def test_ignores_dicts_without_discriminator(self):
        config = {
            "model_id": "chatgpt-4.1",  # no _type
            "nested": {"_type": "other", "model_id": "chatgpt-4.1"},  # wrong _type
        }
        modified = walk_and_replace(config, {"chatgpt-4.1": "chatgpt-4.5"})
        assert modified is False

    def test_recurses_into_nested_structures(self):
        config = {
            "agents": [
                {"config": {"_type": LLM_CONFIG_TYPE, "model_id": "chatgpt-4.1"}}
            ]
        }
        modified = walk_and_replace(config, {"chatgpt-4.1": "chatgpt-4.5"})
        assert modified is True
        assert config["agents"][0]["config"]["model_id"] == "chatgpt-4.5"


class TestMigrateLLMModelsDB:
    """Integration test exercising migrate_llm_models against a real PostgreSQL."""

    @pytest.fixture(autouse=True)
    def _seed_data(self, test_db):
        """Insert test rows into retrieval_agent_generation."""
        self.row_id = str(uuid.uuid4())
        generation_config = {
            "module": "summarize",
            "model": {"_type": LLM_CONFIG_TYPE, "model_id": "chatgpt-4.1"},
            "nested_agents": [
                {
                    "planner_model": {
                        "_type": LLM_CONFIG_TYPE,
                        "model_id": "gemini-2.5-flash",
                    }
                }
            ],
        }
        # We need an agent_id that exists in retrieval_agent_config for the FK.
        # Instead, insert directly bypassing FK by using raw SQL with a temp disable
        # or use a table that has no FK. Let's use retrieval_agents_drivers which
        # also has a JSONB 'config' column.
        # Actually, let's just insert into the table and handle FK by creating
        # a parent row first.
        test_db.execute(
            text(
                "INSERT INTO retrieval_agent_config (account, kbid) "
                "VALUES (:account, :kbid) ON CONFLICT DO NOTHING"
            ),
            {"account": "test-account", "kbid": "test-kb"},
        )
        test_db.execute(
            text(
                "INSERT INTO retrieval_agent_generation (id, account, agent_id, generation) "
                "VALUES (:id, :account, :agent_id, :generation)"
            ),
            {
                "id": self.row_id,
                "account": "test-account",
                "agent_id": "test-kb",
                "generation": json.dumps(generation_config),
            },
        )
        test_db.commit()

    def test_migrate_replaces_models_in_db(self, test_db):
        total = migrate_llm_models(
            test_db,
            {"chatgpt-4.1": "chatgpt-4.5", "gemini-2.5-flash": "gemini-3.0-flash"},
        )
        test_db.commit()

        assert total >= 1

        # Read back and verify
        row = test_db.execute(
            text("SELECT generation FROM retrieval_agent_generation WHERE id = :id"),
            {"id": self.row_id},
        ).fetchone()
        config = row[0] if isinstance(row[0], dict) else json.loads(row[0])

        assert config["model"]["model_id"] == "chatgpt-4.5"
        assert (
            config["nested_agents"][0]["planner_model"]["model_id"]
            == "gemini-3.0-flash"
        )
