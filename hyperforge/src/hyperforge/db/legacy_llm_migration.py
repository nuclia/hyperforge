"""Convert old model strings in stored Hyperforge agent configs to LLMConfig.

Only fields already declared as LLMField in Hyperforge are included. Agent
plugins with string-typed model fields must remain strings until their schemas
and runtimes are migrated separately.
"""

import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

from hyperforge.db.llm_migration_utils import HYPERFORGE_LLM_TABLES
from hyperforge.llm_config import LLM_CONFIG_TYPE

# Keep this inventory aligned with the LLMField declarations in Hyperforge.
# Context fields are inherited by ContextAgentConfig subclasses, including
# the installed progress-agents plugins. Their own string-typed model fields
# are deliberately excluded from MODEL_FIELDS.
CONTEXT_MODULES = {
    "a2a",
    "advanced_ask",
    "ask",
    "basic_ask",
    "brave",
    "context_conditional",
    "cypher",
    "google",
    "http",
    "marklogic",
    "mcp",
    "multi_mcp",
    "pandas",
    "perplexity",
    "perplexity_search",
    "restricted",
    "salesforce",
    "smart",
    "snowflake",
    "sparql",
    "sql",
    "static",
    "static_string",
    "sitefinity",
    "sync",
}
CONTEXT_FIELDS = {"context_validation_model", "rephrase_model", "summarize_model"}
MODEL_FIELDS = {
    "advanced_ask": {"generative_model"},
    "ask": {"configuration_model"},
    "basic_ask": {"generative_model"},
    "context_conditional": {"model"},
    "external": {"model"},
    "generate": {"model"},
    "generation_conditional": {"model"},
    "mcp": {"tool_choice_model", "sampling_model"},
    "multi_mcp": {"summarize_model", "tool_choice_model"},
    "post_conditional": {"model"},
    "pre_conditional": {"model"},
    "related": {"model"},
    "rephrase": {"model"},
    "restart": {"model"},
    "restricted": {"decision_model"},
    "salesforce": {"tool_choice_model", "sampling_model"},
    "smart": {"planner_model", "executor_model"},
    "summarize": {"model"},
    "sync": {"generative_model"},
}
NESTED_FIELDS_BY_MODULE = {
    "advanced_generation": {"summarize_config", "data_viz_config"},
    "context_conditional": {"then", "else_"},
    "generation_conditional": {"then", "else_"},
    "multi_mcp": {"configs"},
    "post_conditional": {"then", "else_"},
    "pre_conditional": {"then", "else_"},
    "restricted": {"agents"},
    "smart": {"registered_agents"},
}


def wrap_legacy_llm_fields(config: Any) -> bool:
    """Wrap string-valued LLMFields, including those inside nested agents."""
    changed = False
    if isinstance(config, list):
        for item in config:
            changed = wrap_legacy_llm_fields(item) or changed
    elif isinstance(config, dict) and isinstance(config.get("module"), str):
        module = config["module"]
        fields = MODEL_FIELDS.get(module, set())
        if module in CONTEXT_MODULES:
            fields = fields | CONTEXT_FIELDS
        for field in fields:
            value = config.get(field)
            if isinstance(value, str):
                config[field] = {"_type": LLM_CONFIG_TYPE, "model_id": value}
                changed = True
        nested_fields = NESTED_FIELDS_BY_MODULE.get(module, set())
        if module in CONTEXT_MODULES:
            nested_fields = nested_fields | {"fallback", "next_agent"}
        for field in nested_fields:
            if field in config:
                changed = wrap_legacy_llm_fields(config[field]) or changed
    return changed


def migrate_legacy_llm_fields(conn: Connection) -> int:
    """Convert stored agent configs, leaving unrelated and structured data alone."""
    modified = 0
    for table, column in HYPERFORGE_LLM_TABLES:
        last_id = "00000000-0000-0000-0000-000000000000"
        while rows := conn.execute(
            text(
                f"SELECT id, {column} FROM {table} "
                "WHERE id > CAST(:last_id AS uuid) ORDER BY id LIMIT 500"
            ),
            {"last_id": last_id},
        ).fetchall():
            for row_id, config in rows:
                if isinstance(config, str):
                    config = json.loads(config)
                if wrap_legacy_llm_fields(config):
                    conn.execute(
                        text(
                            f"UPDATE {table} SET {column} = CAST(:config AS jsonb) "
                            "WHERE id = :id"
                        ),
                        {"config": json.dumps(config), "id": row_id},
                    )
                    modified += 1
            last_id = str(rows[-1][0])
    return modified
