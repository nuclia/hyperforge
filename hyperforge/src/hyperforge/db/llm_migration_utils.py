"""Utilities for migrating LLMConfig model identifiers in stored JSONB configs.

This module provides helpers for Alembic data migrations that need to rename
model identifiers across all stored agent configurations. It recursively walks
JSONB structures and only modifies objects carrying the `_type: "llm_config"`
discriminator, making it safe for arbitrarily nested configs.

Usage in Alembic migrations:

    from hyperforge.db.llm_migration_utils import migrate_llm_models

    def upgrade():
        migrate_llm_models(op.get_bind(), {
            "chatgpt-4.1": "chatgpt-4.5",
            "gemini-2.5-flash": "gemini-3.0-flash",
        })
"""

import json
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

from hyperforge.llm_config import LLM_CONFIG_TYPE

# Tables and JSONB columns known to contain agent configurations with LLMConfig
# instances. Update this list when new tables are added.
HYPERFORGE_LLM_TABLES: list[tuple[str, str]] = [
    ("retrieval_agent_preprocess", "preprocess"),
    ("retrieval_agent_context", "context"),
    ("retrieval_agent_generation", "generation"),
    ("retrieval_agent_postprocess", "postprocess"),
    ("retrieval_agents_drivers", "config"),
]


def walk_and_replace(obj: Any, replacements: dict[str, str]) -> bool:
    """Recursively walk a JSON structure, replacing model IDs only in LLMConfig objects.

    Handles the {model}/{uuid} suffix format by matching on the base model ID
    and preserving the UUID suffix.

    Args:
        obj: A deserialized JSON structure (dict, list, or scalar).
        replacements: Mapping of old_model_id -> new_model_id.

    Returns:
        True if any modification was made.
    """
    modified = False
    if isinstance(obj, dict):
        if obj.get("_type") == LLM_CONFIG_TYPE and "model_id" in obj:
            model_value = obj["model_id"]
            base_model = model_value.split("/")[0]
            if base_model in replacements:
                # Preserve the /<uuid> suffix if present
                suffix = model_value[len(base_model) :]  # "" or "/<uuid>"
                obj["model_id"] = replacements[base_model] + suffix
                modified = True
        # Recurse into all values regardless (there may be nested LLMConfigs)
        for value in obj.values():
            if walk_and_replace(value, replacements):
                modified = True
    elif isinstance(obj, list):
        for item in obj:
            if walk_and_replace(item, replacements):
                modified = True
    return modified


def migrate_llm_models(conn: Connection, replacements: dict[str, str]) -> int:
    """Replace model identifiers in all stored LLMConfig instances.

    Recursively walks stored JSONB configs and only modifies objects
    carrying the _type="llm_config" discriminator. Safe for arbitrarily
    nested structures and multi-config columns.

    Handles the {model}/{uuid} format automatically - only the base model
    is matched and replaced, the UUID suffix is preserved.

    Args:
        conn: Database connection (from op.get_bind()).
        replacements: Mapping of old_model_id -> new_model_id.

    Returns:
        Number of rows modified.
    """
    total_modified = 0
    for table, column in HYPERFORGE_LLM_TABLES:
        # Only fetch rows that contain at least one LLMConfig (pre-filter)
        rows = conn.execute(
            text(
                f"SELECT id, {column} FROM {table} "
                f'WHERE {column}::text LIKE \'%"_type": "{LLM_CONFIG_TYPE}"%\''
            )
        ).fetchall()
        for row_id, config in rows:
            if config is None:
                continue
            if walk_and_replace(config, replacements):
                conn.execute(
                    text(f"UPDATE {table} SET {column} = :config WHERE id = :id"),
                    {"config": json.dumps(config), "id": row_id},
                )
                total_modified += 1
    return total_modified
