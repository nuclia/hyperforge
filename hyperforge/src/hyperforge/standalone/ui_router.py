"""
UI API router for the standalone ARAG deployment.

Provides endpoints consumed exclusively by the built-in frontend:
  GET  /api/v1/ui/schema   — all registered agent/driver schemas
  GET  /api/v1/ui/config   — current in-memory agent config (full JSON)
  PUT  /api/v1/ui/config   — replace config, persist to AGENTS_CONFIG file
  GET  /api/v1/ui/models   — list of known model IDs for model_select widget

No authentication is required (the OpenAuthBackend already grants all roles
in standalone mode, and the UI is local-only by design).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from hyperforge.standalone.config import StandaloneConfig

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Schema — expose the registered agent/driver registry so the UI can build
# the "add agent" palette and the config form for each agent type.
# ---------------------------------------------------------------------------


# Properties that link to other agents (subagents). The frontend renders these
# as edges/child nodes on the canvas instead of inline form fields.
CONNECTABLE_KEYS: list[str] = [
    "fallback",
    "next_agent",
    "then",
    "else_",
    "agents",
    "registered_agents",
]


def _merge_defs(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Merge `source["$defs"]` into target without overwriting existing keys."""
    defs = source.get("$defs") or {}
    for name, schema in defs.items():
        target.setdefault(name, schema)


def _module_to_def_name(config_schema: dict[str, Any]) -> str | None:
    """Return the $defs key name for an agent config schema (= its `title`)."""
    return config_schema.get("title")


@router.get("/api/v1/ui/schema", response_class=JSONResponse)
async def get_schema() -> dict[str, Any]:
    """Return all registered agent and driver schemas plus a merged $defs map.

    Payload shape:
        {
          "preprocess": { module_id: AgentSchema, ... },
          "context":    { ... },
          "generation": { ... },
          "postprocess":{ ... },
          "drivers":    { provider_id: DriverSchema, ... },
          "$defs":      { ClassName: JsonSchema, ... },   # merged from all agents
          "connectable_keys": [...],
          "agent_module_to_def": { module_id: ClassName, ... }
        }
    """
    from hyperforge.configure import (
        get_context_agent_schemas,
        get_driver_agent_schemas,
        get_generation_agent_schemas,
        get_postprocess_agent_schemas,
        get_preprocess_agent_schemas,
    )

    stages = {
        "preprocess": get_preprocess_agent_schemas(),
        "context": get_context_agent_schemas(),
        "generation": get_generation_agent_schemas(),
        "postprocess": get_postprocess_agent_schemas(),
    }
    drivers = get_driver_agent_schemas()

    # Merge all $defs into a single top-level map so the frontend can resolve
    # `$ref: "#/$defs/<ClassName>"` across stages/drivers.
    merged_defs: dict[str, Any] = {}
    agent_module_to_def: dict[str, str] = {}

    for stage_agents in stages.values():
        for module_id, agent_schema in stage_agents.items():
            cfg = agent_schema.get("config_schema") or {}
            _merge_defs(merged_defs, cfg)
            # The top-level config schema itself is also exposed under $defs so
            # nested oneOf/discriminator chains can resolve it.
            def_name = _module_to_def_name(cfg)
            if def_name:
                merged_defs.setdefault(def_name, cfg)
                agent_module_to_def[module_id] = def_name

    for driver_schema in drivers.values():
        cfg = driver_schema.get("config_schema") or {}
        _merge_defs(merged_defs, cfg)

    return {
        **stages,
        "drivers": drivers,
        "$defs": merged_defs,
        "connectable_keys": CONNECTABLE_KEYS,
        "agent_module_to_def": agent_module_to_def,
    }


# ---------------------------------------------------------------------------
# Config — read / write the full agent config
# ---------------------------------------------------------------------------


@router.get("/api/v1/ui/config", response_class=JSONResponse)
async def get_config(request: Request) -> dict[str, Any]:
    """Return the current in-memory config as a JSON-serialisable dict."""
    agent_manager = request.app.agent_manager
    return {
        agent_id: agent_cfg.model_dump(mode="json")
        for agent_id, agent_cfg in agent_manager._config.items()
    }


@router.put("/api/v1/ui/config", response_class=JSONResponse)
async def put_config(request: Request) -> dict[str, Any]:
    """
    Replace the in-memory config and persist it to the config file on disk.

    The request body must be a JSON object whose structure matches the
    standalone config format (same as AGENTS_CONFIG).
    """
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc

    # Validate through the Pydantic model so we get clear error messages.
    try:
        new_config = StandaloneConfig.validate_python(body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Update the live in-memory config.
    agent_manager = request.app.agent_manager
    agent_manager._config = new_config

    # Persist to disk so the change survives a restart.
    settings = request.app._standalone_settings
    config_path = settings.agents_config
    try:
        serialised = {
            agent_id: agent_cfg.model_dump(mode="json")
            for agent_id, agent_cfg in new_config.items()
        }
        if config_path.suffix in (".yaml", ".yml"):
            config_path.write_text(yaml.dump(serialised, allow_unicode=True))
        else:
            config_path.write_text(json.dumps(serialised, indent=2, ensure_ascii=False))
        logger.info("Config persisted to %s", config_path)
    except Exception as exc:
        # Non-fatal — in-memory update already succeeded.
        logger.warning("Could not persist config to %s: %s", config_path, exc)

    return {"status": "ok", "agents": list(new_config.keys())}


# ---------------------------------------------------------------------------
# Models — return a list of known model IDs for the model_select widget.
# Collects defaults from all registered agent config schemas where the field
# carries widget="model_select", then merges with a hard-coded baseline list.
# ---------------------------------------------------------------------------

_BASELINE_MODELS: list[str] = [
    "chatgpt-azure-4o-mini",
    "chatgpt-azure-4o",
    "chatgpt4o",
    "chatgpt-4.1",
    "chatgpt-5",
    "chatgpt-o3-mini",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-flash-preview",
    "claude-4-5-haiku",
    "claude-4-5-sonnet",
    "gcp-claude-4-5-haiku",
    "gcp-claude-4-5-sonnet",
]


def _collect_model_defaults() -> list[str]:
    """Walk all agent config schemas and harvest model field defaults."""
    from hyperforge.configure import (
        get_context_agent_schemas,
        get_driver_agent_schemas,
        get_generation_agent_schemas,
        get_postprocess_agent_schemas,
        get_preprocess_agent_schemas,
    )

    seen: set[str] = set(_BASELINE_MODELS)

    all_schemas = [
        *get_preprocess_agent_schemas().values(),
        *get_context_agent_schemas().values(),
        *get_generation_agent_schemas().values(),
        *get_postprocess_agent_schemas().values(),
        *get_driver_agent_schemas().values(),
    ]

    for agent_schema in all_schemas:
        config_schema = agent_schema.get("config_schema", {})
        for _field_name, field_schema in config_schema.get("properties", {}).items():
            if field_schema.get("widget") == "model_select":
                default = field_schema.get("default")
                if isinstance(default, str) and default:
                    seen.add(default)

    # Return baseline first (preserves a sensible order), then extras
    result = list(_BASELINE_MODELS)
    for m in sorted(seen - set(_BASELINE_MODELS)):
        result.append(m)
    return result


@router.get("/api/v1/ui/models", response_class=JSONResponse)
async def get_models() -> list[str]:
    """Return a sorted list of known model IDs for the model_select widget."""
    return _collect_model_defaults()
