"""Deterministic transpiler and PostgreSQL data migration from 4-stage workflows to Harness."""

from __future__ import annotations

import json
import keyword
import re
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any
from uuid import uuid4

from RestrictedPython import compile_restricted
from sqlalchemy import text
from sqlalchemy.engine import Connection

from hyperforge.llm_config import LLMConfig, llm_defaults

MigrationHook = Callable[[dict[str, Any]], dict[str, Any]]

_BUILTIN_TOOLKIT_MODULES = {
    "a2a",
    "advanced_ask",
    "ask",
    "basic_ask",
    "context_conditional",
    "external",
    "generate",
    "generation_conditional",
    "google",
    "historical",
    "http",
    "mcp",
    "multi_mcp",
    "passthrough",
    "perplexity",
    "perplexity_search",
    "post_conditional",
    "pre_conditional",
    "related",
    "remi",
    "rephrase",
    "restart",
    "restricted",
    "smart",
    "static",
    "static_string",
    "summarize",
    "sync",
}


def _extract_model_id(raw_model: Any, fallback: str) -> str:
    if isinstance(raw_model, str) and raw_model.strip():
        return raw_model.strip()
    if isinstance(raw_model, LLMConfig):
        return raw_model.model_id
    if isinstance(raw_model, dict) and isinstance(raw_model.get("model_id"), str):
        return str(raw_model["model_id"])
    return fallback


def _llm_dict(raw_model: Any, fallback: str) -> dict[str, Any]:
    if isinstance(raw_model, LLMConfig):
        return raw_model.model_dump()
    if isinstance(raw_model, dict) and "model_id" in raw_model:
        return dict(raw_model)
    if isinstance(raw_model, str) and raw_model.strip():
        return LLMConfig(model_id=raw_model.strip()).model_dump()
    return LLMConfig(model_id=fallback).model_dump()


class _NamespaceAllocator:
    def __init__(self) -> None:
        self._used: set[str] = set()
        self._counters: dict[str, int] = {}

    def allocate(self, hint: str) -> str:
        base = re.sub(r"[^a-zA-Z0-9_]", "_", hint).strip("_").lower() or "tk"
        base = base.replace("__", "_")
        if base[0].isdigit() or keyword.iskeyword(base):
            base = f"tk_{base}"
        if base not in self._used:
            self._used.add(base)
            return base
        idx = self._counters.get(base, 1) + 1
        while f"{base}_{idx}" in self._used:
            idx += 1
        self._counters[base] = idx
        candidate = f"{base}_{idx}"
        self._used.add(candidate)
        return candidate


class _WorkflowCompiler:
    def __init__(
        self,
        external_migrations: Mapping[str, MigrationHook] | None = None,
    ) -> None:
        self.external_migrations = dict(external_migrations or {})
        self.ns_alloc = _NamespaceAllocator()
        self.toolkits: list[dict[str, Any]] = []
        self._shared_ns: dict[str, str] = {}

    def _ensure_shared_toolkit(self, key: str, toolkit_dict: dict[str, Any]) -> str:
        if key in self._shared_ns:
            return self._shared_ns[key]
        ns = self.ns_alloc.allocate(key)
        entry = {
            "id": str(uuid4()),
            **toolkit_dict,
            "namespace": ns,
            "ingest_skills": True,
        }
        self.toolkits.append(entry)
        self._shared_ns[key] = ns
        return ns

    def register_agent_as_toolkit(
        self,
        agent_cfg: dict[str, Any],
        *,
        lazy_load: bool = False,
    ) -> tuple[str, str]:
        """Convert a single legacy agent config dict into a toolkit dict and return `(ns, primary_call)`."""
        module = str(agent_cfg.get("module") or "unknown")
        hint = str(agent_cfg.get("id") or module)
        if len(hint) > 24 or "-" in hint:
            hint = module

        if module in self.external_migrations:
            converted = deepcopy(self.external_migrations[module](agent_cfg))
            ns = self.ns_alloc.allocate(str(converted.get("namespace") or hint))
            converted.setdefault("id", str(uuid4()))
            converted["namespace"] = ns
            converted.setdefault("lazy_load", lazy_load)
            converted.setdefault("ingest_skills", True)
            primary_fn = str(converted.pop("_primary_tool", "query"))
            self.toolkits.append(converted)
            return ns, f"{ns}__{primary_fn}"

        ns = self.ns_alloc.allocate(hint)
        title = agent_cfg.get("title")
        base_fields = {
            "id": str(uuid4()),
            "namespace": ns,
            "title": title if isinstance(title, str) else None,
            "lazy_load": lazy_load,
            "ingest_skills": True,
        }

        if module in {"basic_ask", "ask"}:
            self.toolkits.append(
                {
                    **base_fields,
                    "toolkit": "nucliadb",
                    "sources": list(agent_cfg.get("sources") or []),
                    "generative_model": _llm_dict(
                        agent_cfg.get("generative_model"), llm_defaults.default
                    ),
                }
            )
            return ns, f"{ns}__ask"

        if module == "advanced_ask":
            self.toolkits.append(
                {
                    **base_fields,
                    "toolkit": "nucliadb",
                    "sources": list(agent_cfg.get("sources") or []),
                    "generative_model": _llm_dict(
                        agent_cfg.get("generative_model"), llm_defaults.default
                    ),
                }
            )
            return ns, f"{ns}__advanced_ask"

        if module == "sync":
            self.toolkits.append(
                {
                    **base_fields,
                    "toolkit": "nucliadb_sync",
                    "sources": list(agent_cfg.get("sources") or []),
                    "generative_model": _llm_dict(
                        agent_cfg.get("generative_model"), llm_defaults.default
                    ),
                }
            )
            return ns, f"{ns}__ask"

        if module == "mcp":
            self.toolkits.append(
                {
                    **base_fields,
                    "toolkit": "mcp",
                    "source": str(agent_cfg.get("source") or "mcp"),
                    "transport": str(agent_cfg.get("transport") or "HTTP"),
                    "valid_headers": list(agent_cfg.get("valid_headers") or []),
                    "read_timeout_seconds": int(
                        agent_cfg.get("read_timeout_seconds") or 400
                    ),
                    "roots": dict(agent_cfg.get("roots") or {}),
                    "sampling_model": _llm_dict(
                        agent_cfg.get("sampling_model"), llm_defaults.fast
                    ),
                }
            )
            return ns, ""

        if module == "a2a":
            self.toolkits.append(
                {
                    **base_fields,
                    "toolkit": "a2a",
                    "source": str(agent_cfg.get("source") or "a2a"),
                    "remote_workflow_id": str(
                        agent_cfg.get("remote_workflow_id") or "default"
                    ),
                    "extra_metadata": dict(agent_cfg.get("extra_metadata") or {}),
                    "valid_headers": list(agent_cfg.get("valid_headers") or []),
                }
            )
            return ns, f"{ns}__a2a_send"

        if module == "google":
            self.toolkits.append(
                {
                    **base_fields,
                    "toolkit": "google",
                    "source": str(agent_cfg.get("source") or "google"),
                    "gen_model_id": str(
                        agent_cfg.get("gen_model_id") or "gemini-3.6-flash"
                    ),
                }
            )
            return ns, f"{ns}__internet_search"

        if module in {"perplexity", "perplexity_search"}:
            self.toolkits.append(
                {
                    **base_fields,
                    "toolkit": "perplexity",
                    "source": str(agent_cfg.get("source") or "perplexity"),
                    "domain": list(agent_cfg.get("domain") or []),
                    "search_context_size": str(
                        agent_cfg.get("search_context_size") or "low"
                    ),
                    "images": bool(agent_cfg.get("images", False)),
                    "prompt": agent_cfg.get("prompt"),
                    "max_results": int(agent_cfg.get("max_results") or 10),
                    "max_tokens_per_page": int(
                        agent_cfg.get("max_tokens_per_page") or 4096
                    ),
                }
            )
            fn = (
                "perplexity_search"
                if module == "perplexity_search"
                else "perplexity_answer"
            )
            return ns, f"{ns}__{fn}"

        if module == "http":
            self.toolkits.append(
                {
                    **base_fields,
                    "toolkit": "http",
                    "method": str(agent_cfg.get("method") or "GET"),
                    "url": str(agent_cfg.get("url") or "https://example.com"),
                    "headers": dict(agent_cfg.get("headers") or {}),
                    "question_query_param": agent_cfg.get("question_query_param"),
                    "question_post_field": agent_cfg.get("question_post_field"),
                    "timeout": int(agent_cfg.get("timeout") or 10),
                    "max_response_bytes": int(
                        agent_cfg.get("max_response_bytes") or 1048576
                    ),
                }
            )
            return ns, f"{ns}__http_fetch"

        if module == "external":
            self.toolkits.append(
                {
                    **base_fields,
                    "toolkit": "external",
                    "url": str(agent_cfg.get("url") or "https://example.com"),
                    "method": str(agent_cfg.get("method") or "POST"),
                    "headers": dict(agent_cfg.get("headers") or {}),
                    "prompt": agent_cfg.get("prompt"),
                    "description": agent_cfg.get("description"),
                    "call_schema": agent_cfg.get("call_schema"),
                    "call_obj": agent_cfg.get("call_obj"),
                    "model": _llm_dict(agent_cfg.get("model"), llm_defaults.reasoning),
                    "timeout": int(agent_cfg.get("timeout") or 30),
                    "max_response_bytes": int(
                        agent_cfg.get("max_response_bytes") or 1048576
                    ),
                }
            )
            return ns, f"{ns}__webhook_extract_and_send"

        if module in {"static", "static_string"}:
            self.toolkits.append(
                {
                    **base_fields,
                    "toolkit": "static",
                    "context": str(agent_cfg.get("context") or ""),
                    "structured": agent_cfg.get("structured"),
                }
            )
            return ns, f"{ns}__get_static_context"

        if module == "rephrase":
            self.toolkits.append(
                {
                    **base_fields,
                    "toolkit": "rephrase",
                    "model": _llm_dict(agent_cfg.get("model"), llm_defaults.default),
                    "rules": list(agent_cfg.get("rules") or []),
                    "provided_synonyms": dict(agent_cfg.get("provided_synonyms") or {}),
                    "history": bool(agent_cfg.get("history", True)),
                }
            )
            fn = "split_query" if agent_cfg.get("split_question") else "rephrase_query"
            return ns, f"{ns}__{fn}"

        if module in {"summarize", "generate", "passthrough"}:
            self.toolkits.append(
                {
                    **base_fields,
                    "toolkit": "synthesis",
                    "model": _llm_dict(agent_cfg.get("model"), llm_defaults.default),
                    "system_prompt": agent_cfg.get("system_prompt"),
                    "prompt": agent_cfg.get("prompt"),
                    "citations": bool(agent_cfg.get("citations", False)),
                    "force_chunk_level_citations": bool(
                        agent_cfg.get("force_chunk_level_citations", False)
                    ),
                    "conversational": bool(agent_cfg.get("conversational", False)),
                    "rich_context": bool(agent_cfg.get("rich_context", False)),
                    "rules": list(agent_cfg.get("rules") or []),
                }
            )
            fn = (
                "synthesize_answer"
                if module == "summarize"
                else "generate_text"
                if module == "generate"
                else "passthrough_contexts"
            )
            return ns, f"{ns}__{fn}"

        # Generic pass-through for external third-party plugin modules
        # (e.g., snowflake, salesforce, sql, cypher, sparql, marklogic, sitefinity, brave, pandas)
        preserved = {
            k: deepcopy(v)
            for k, v in agent_cfg.items()
            if k
            not in {
                "id",
                "module",
                "title",
                "fallback",
                "next_agent",
                "published_functions",
            }
        }
        self.toolkits.append(
            {
                **base_fields,
                "toolkit": module,
                **preserved,
            }
        )
        pub = agent_cfg.get("published_functions")
        primary = str(pub[0]) if isinstance(pub, (list, tuple)) and pub else "query"
        return ns, f"{ns}__{primary}"

    def _emit_preprocess_lines(
        self, items: list[dict[str, Any]], indent: str = ""
    ) -> list[str]:
        lines: list[str] = []
        for item in items:
            module = str(item.get("module") or "")
            if module == "pre_conditional":
                dec_ns = self._ensure_shared_toolkit(
                    "decision",
                    {
                        "toolkit": "decision",
                        "model": _llm_dict(item.get("model"), llm_defaults.reasoning),
                    },
                )
                cond_prompt = json.dumps(str(item.get("prompt") or ""))
                sim = json.dumps(list(item.get("similarity") or []))
                lines.append(
                    f"{indent}wf_cond = {dec_ns}__check_condition(text=question, condition_prompt={cond_prompt}, similarity={sim})"
                )
                lines.append(f"{indent}if wf_cond['matched']:")
                then_lines = self._emit_preprocess_lines(
                    list(item.get("then") or []), indent + "    "
                )
                lines.extend(then_lines or [f"{indent}    pass"])
                else_items = list(item.get("else_") or [])
                if else_items:
                    lines.append(f"{indent}else:")
                    else_lines = self._emit_preprocess_lines(
                        else_items, indent + "    "
                    )
                    lines.extend(else_lines or [f"{indent}    pass"])
            elif module == "rephrase":
                _, call_fn = self.register_agent_as_toolkit(item)
                lines.append(f"{indent}wf_prep = {call_fn}(question=question)")
                lines.append(f"{indent}queries = wf_prep['questions']")
            elif module == "historical":
                continue
            else:
                _, call_fn = self.register_agent_as_toolkit(item)
                if call_fn:
                    lines.append(f"{indent}{call_fn}(question=question)")
        return lines

    def _emit_context_agent_lines(
        self,
        item: dict[str, Any],
        query_var: str,
        target_list: str,
        indent: str = "",
    ) -> list[str]:
        lines: list[str] = []
        module = str(item.get("module") or "")

        if module == "context_conditional":
            dec_ns = self._ensure_shared_toolkit(
                "decision",
                {
                    "toolkit": "decision",
                    "model": _llm_dict(item.get("model"), llm_defaults.reasoning),
                },
            )
            cond_prompt = json.dumps(str(item.get("prompt") or ""))
            cond_fallback = item.get("fallback")
            cond_next = item.get("next_agent")
            has_cond_chain = isinstance(cond_fallback, dict) or isinstance(
                cond_next, dict
            )
            branch_target = "wf_cond_ctxs" if has_cond_chain else target_list
            if has_cond_chain:
                lines.append(f"{indent}wf_cond_ctxs = []")
            lines.append(
                f"{indent}wf_cond = {dec_ns}__check_condition(text={query_var}, condition_prompt={cond_prompt})"
            )
            lines.append(f"{indent}if wf_cond['matched']:")
            then_items = list(item.get("then") or [])
            then_body: list[str] = []
            for sub in then_items:
                then_body.extend(
                    self._emit_context_agent_lines(
                        sub, query_var, branch_target, indent + "    "
                    )
                )
            lines.extend(then_body or [f"{indent}    pass"])
            else_items = list(item.get("else_") or [])
            if else_items:
                lines.append(f"{indent}else:")
                else_body: list[str] = []
                for sub in else_items:
                    else_body.extend(
                        self._emit_context_agent_lines(
                            sub, query_var, branch_target, indent + "    "
                        )
                    )
                lines.extend(else_body or [f"{indent}    pass"])
            if has_cond_chain:
                ctxops_ns = self._ensure_shared_toolkit(
                    "context_ops",
                    {
                        "toolkit": "context_ops",
                        "validation_model": _llm_dict(
                            item.get("context_validation_model"), llm_defaults.default
                        ),
                        "rephrase_model": _llm_dict(
                            item.get("rephrase_model"), llm_defaults.default
                        ),
                    },
                )
                lines.append(
                    f"{indent}wf_cond_in = wf_cond_ctxs or [{{'question': {query_var}, 'source': 'conditional', 'agent': 'Conditional', 'chunks': [{{'chunk_id': 'empty', 'text': ''}}]}}]"
                )
                lines.append(
                    f"{indent}wf_val = {ctxops_ns}__validate_contexts(question={query_var}, contexts=wf_cond_in, prune=False)"
                )
                lines.append(f"{indent}if wf_cond_ctxs:")
                lines.append(f"{indent}    {target_list}.extend(wf_val['contexts'])")
                if isinstance(cond_fallback, dict):
                    lines.append(
                        f"{indent}wf_fb_q = wf_val['missing_query'] or ('' if wf_val['useful'] else {query_var})"
                    )
                    lines.append(f"{indent}if wf_fb_q:")
                    lines.extend(
                        self._emit_context_agent_lines(
                            cond_fallback,
                            "wf_fb_q",
                            target_list,
                            indent + "    ",
                        )
                    )
                if isinstance(cond_next, dict):
                    lines.append(
                        f"{indent}wf_summaries = [c.get('summary', '') for c in wf_val['contexts'] if c.get('summary')]"
                    )
                    lines.append(
                        f"{indent}wf_next_q = {ctxops_ns}__rephrase_for_next(question={query_var}, summaries=wf_summaries)"
                    )
                    lines.extend(
                        self._emit_context_agent_lines(
                            cond_next,
                            "wf_next_q['question']",
                            target_list,
                            indent,
                        )
                    )
            return lines

        if module == "smart":
            for child in item.get("registered_agents") or []:
                if isinstance(child, dict):
                    lines.extend(
                        self._emit_context_agent_lines(
                            child, query_var, target_list, indent
                        )
                    )
            return lines

        if module == "multi_mcp":
            for mcp_cfg in item.get("configs") or []:
                if isinstance(mcp_cfg, dict):
                    self.register_agent_as_toolkit(mcp_cfg, lazy_load=True)
            return lines

        if module == "restricted":
            for child in item.get("agents") or []:
                if isinstance(child, dict):
                    self.register_agent_as_toolkit(child)
            raw_code = str(item.get("code") or "").strip()
            if raw_code:
                for raw_line in raw_code.splitlines():
                    if not raw_line.strip().startswith("save("):
                        lines.append(f"{indent}{raw_line}")
            return lines

        _, call_fn = self.register_agent_as_toolkit(item)
        if not call_fn:
            return lines

        if module in {"static", "static_string"}:
            lines.append(f"{indent}wf_res = {call_fn}()")
        elif module == "advanced_ask":
            kwargs_parts = [f"question={query_var}"]
            if item.get("top_k") is not None:
                kwargs_parts.append(f"top_k={int(item['top_k'])}")
            if item.get("min_score") is not None and isinstance(
                item["min_score"], (int, float)
            ):
                kwargs_parts.append(f"min_score={float(item['min_score'])}")
            if item.get("generate_answer") is False:
                kwargs_parts.append("generate_answer=False")
            lines.append(f"{indent}wf_res = {call_fn}({', '.join(kwargs_parts)})")
        else:
            lines.append(f"{indent}wf_res = {call_fn}(question={query_var})")

        lines.append(f"{indent}wf_stage_ctxs = wf_res.get('contexts', [])")

        prune = bool(item.get("prune_context", True))
        fallback_cfg = item.get("fallback")
        next_cfg = item.get("next_agent")
        needs_validation = (
            prune or isinstance(fallback_cfg, dict) or isinstance(next_cfg, dict)
        )
        if needs_validation:
            ctxops_ns = self._ensure_shared_toolkit(
                "context_ops",
                {
                    "toolkit": "context_ops",
                    "validation_model": _llm_dict(
                        item.get("context_validation_model")
                        or item.get("summarize_model"),
                        llm_defaults.default,
                    ),
                    "rephrase_model": _llm_dict(
                        item.get("rephrase_model"), llm_defaults.default
                    ),
                },
            )
            lines.append(
                f"{indent}wf_val = {ctxops_ns}__validate_contexts(question={query_var}, contexts=wf_stage_ctxs, prune={prune})"
            )
            lines.append(f"{indent}{target_list}.extend(wf_val['contexts'])")
            if isinstance(fallback_cfg, dict):
                lines.append(f"{indent}if wf_val['missing_query']:")
                lines.extend(
                    self._emit_context_agent_lines(
                        fallback_cfg,
                        "wf_val['missing_query']",
                        target_list,
                        indent + "    ",
                    )
                )
            if isinstance(next_cfg, dict):
                lines.append(
                    f"{indent}wf_summaries = [c.get('summary', '') for c in wf_val['contexts'] if c.get('summary')]"
                )
                lines.append(
                    f"{indent}wf_next_q = {ctxops_ns}__rephrase_for_next(question={query_var}, summaries=wf_summaries)"
                )
                lines.extend(
                    self._emit_context_agent_lines(
                        next_cfg,
                        "wf_next_q['question']",
                        target_list,
                        indent,
                    )
                )
        else:
            lines.append(f"{indent}{target_list}.extend(wf_stage_ctxs)")

        return lines

    def _emit_generation_lines(
        self, items: list[dict[str, Any]], indent: str = ""
    ) -> list[str]:
        lines: list[str] = []
        if not items:
            return lines

        for item in items:
            module = str(item.get("module") or "")
            if module == "generation_conditional":
                dec_ns = self._ensure_shared_toolkit(
                    "decision",
                    {
                        "toolkit": "decision",
                        "model": _llm_dict(item.get("model"), llm_defaults.reasoning),
                    },
                )
                cond_prompt = json.dumps(str(item.get("prompt") or ""))
                lines.append(
                    f"{indent}wf_cond = {dec_ns}__check_condition(text=question, condition_prompt={cond_prompt})"
                )
                lines.append(f"{indent}if wf_cond['matched']:")
                then_lines = self._emit_generation_lines(
                    list(item.get("then") or []), indent + "    "
                )
                lines.extend(then_lines or [f"{indent}    pass"])
                else_items = list(item.get("else_") or [])
                if else_items:
                    lines.append(f"{indent}else:")
                    else_lines = self._emit_generation_lines(
                        else_items, indent + "    "
                    )
                    lines.extend(else_lines or [f"{indent}    pass"])
            elif module == "summarize":
                _, call_fn = self.register_agent_as_toolkit(item)
                lines.append(
                    f"{indent}wf_gen = {call_fn}(question=question, contexts=contexts)"
                )
                lines.append(f"{indent}final_answer = wf_gen['answer']")
                lines.append(f"{indent}final_citations = wf_gen['citations']")
                lines.append(f"{indent}is_answered = wf_gen['is_answered']")
            elif module == "generate":
                _, call_fn = self.register_agent_as_toolkit(item)
                lines.append(
                    f"{indent}wf_gen = {call_fn}(question=question, contexts=contexts)"
                )
                lines.append(f"{indent}final_answer = wf_gen['text']")
                lines.append(f"{indent}is_answered = True")
            elif module == "passthrough":
                _, call_fn = self.register_agent_as_toolkit(item)
                lines.append(f"{indent}wf_gen = {call_fn}(contexts=contexts)")
                lines.append(f"{indent}final_answer = wf_gen['text']")
                lines.append(f"{indent}if wf_gen.get('rich_answer') is not None:")
                lines.append(
                    f"{indent}    metadata['rich_answer'] = wf_gen['rich_answer']"
                )
                lines.append(f"{indent}is_answered = True")
            else:
                _, call_fn = self.register_agent_as_toolkit(item)
                if call_fn:
                    lines.append(
                        f"{indent}wf_gen = {call_fn}(question=question, contexts=contexts)"
                    )
                    lines.append(
                        f"{indent}final_answer = wf_gen.get('answer', str(wf_gen))"
                    )
        return lines

    def _emit_postprocess_lines(
        self, items: list[dict[str, Any]], indent: str = ""
    ) -> list[str]:
        lines: list[str] = []
        for item in items:
            module = str(item.get("module") or "")
            if module == "restart":
                continue
            if module == "post_conditional":
                dec_ns = self._ensure_shared_toolkit(
                    "decision",
                    {
                        "toolkit": "decision",
                        "model": _llm_dict(item.get("model"), llm_defaults.reasoning),
                    },
                )
                cond_prompt = json.dumps(str(item.get("prompt") or ""))
                subject = "final_answer" if item.get("on") == "ANSWER" else "question"
                lines.append(
                    f"{indent}wf_cond = {dec_ns}__check_condition(text={subject}, condition_prompt={cond_prompt})"
                )
                lines.append(f"{indent}if wf_cond['matched']:")
                then_lines = self._emit_postprocess_lines(
                    list(item.get("then") or []), indent + "    "
                )
                lines.extend(then_lines or [f"{indent}    pass"])
                else_items = list(item.get("else_") or [])
                if else_items:
                    lines.append(f"{indent}else:")
                    else_lines = self._emit_postprocess_lines(
                        else_items, indent + "    "
                    )
                    lines.extend(else_lines or [f"{indent}    pass"])
            elif module == "remi":
                raw_gran = item.get("context_granularity")
                gran_val = str(getattr(raw_gran, "value", raw_gran) or "full")
                eval_ns = self._ensure_shared_toolkit(
                    f"evaluation_{gran_val}",
                    {
                        "toolkit": "evaluation",
                        "context_granularity": gran_val,
                    },
                )
                lines.append(
                    f"{indent}wf_remi = {eval_ns}__evaluate_remi(question=question, answer=final_answer, contexts=contexts, is_answered=is_answered)"
                )
                lines.append(f"{indent}metadata['remi'] = wf_remi")
                lines.append(f"{indent}contexts.extend(wf_remi.get('contexts', []))")
            elif module == "related":
                eval_ns = self._ensure_shared_toolkit(
                    "evaluation",
                    {
                        "toolkit": "evaluation",
                        "model": _llm_dict(item.get("model"), llm_defaults.default),
                        "related_prompt": item.get("prompt"),
                    },
                )
                lines.append(
                    f"{indent}wf_rel = {eval_ns}__suggest_related_questions(question=question, contexts=contexts)"
                )
                lines.append(f"{indent}related_questions.extend(wf_rel['questions'])")
            elif module == "external":
                _, call_fn = self.register_agent_as_toolkit(item)
                if item.get("context"):
                    ns = call_fn.split("__", 1)[0]
                    lines.append(f"{indent}{ns}__webhook_send(payload=contexts)")
                else:
                    lines.append(
                        f"{indent}{call_fn}(question=question, answer=final_answer, contexts=contexts)"
                    )
            else:
                _, call_fn = self.register_agent_as_toolkit(item)
                if call_fn:
                    lines.append(f"{indent}{call_fn}(question=question)")
        return lines


def _is_pure_loop_workflow(
    preprocess: list[dict[str, Any]],
    context: list[dict[str, Any]],
    generation: list[dict[str, Any]],
    postprocess: list[dict[str, Any]],
) -> bool:
    if preprocess or postprocess:
        return False
    if len(generation) > 1:
        return False
    if generation and generation[0].get("module") not in {"summarize"}:
        return False
    if not context:
        return True
    if len(context) == 1:
        ctx = context[0]
        if ctx.get("fallback") or ctx.get("next_agent"):
            return False
        return ctx.get("module") in {"smart", "mcp", "multi_mcp"}
    return False


def transpile_stages_to_harness(
    *,
    preprocess: list[dict[str, Any]],
    context: list[dict[str, Any]],
    generation: list[dict[str, Any]],
    postprocess: list[dict[str, Any]],
    external_migrations: Mapping[str, MigrationHook] | None = None,
) -> dict[str, Any]:
    """Deterministically convert 4-stage JSON configs into unified Harness parts.

    Returns a dictionary containing:
    - `execution_mode`: `'loop'` or `'code'`
    - `harness_config`: JSONB settings dict for `retrieval_agent_workflow.harness_config`
    - `workflow_code`: RestrictedPython source string (or `None` when `execution_mode == 'loop'`)
    - `toolkits`: list of toolkit dicts ready for `agent_workflow_toolkits` or `HarnessAgentConfig`
    """
    compiler = _WorkflowCompiler(external_migrations=external_migrations)

    if _is_pure_loop_workflow(preprocess, context, generation, postprocess):
        model_id = llm_defaults.smart
        feedback_enabled = False
        max_turns: int | None = None
        prompt_parts: list[str] = []
        enabled_skills: list[str] = []

        if context:
            ctx = context[0]
            mod = ctx.get("module")
            if mod == "smart":
                model_id = _extract_model_id(
                    ctx.get("executor_model"), llm_defaults.smart
                )
                feedback_enabled = bool(ctx.get("enable_user_feedback", False))
                max_turns = int(ctx.get("max_iterations") or 5) * 2
                if ctx.get("planning_mode") == "plan_execute":
                    enabled_skills.append("plan_and_execute")
                if ctx.get("extra_prompt"):
                    prompt_parts.append(str(ctx["extra_prompt"]))
                for reg in ctx.get("registered_agents") or []:
                    if isinstance(reg, dict):
                        if reg.get("module") == "multi_mcp":
                            for m_cfg in reg.get("configs") or []:
                                if isinstance(m_cfg, dict):
                                    compiler.register_agent_as_toolkit(
                                        m_cfg, lazy_load=True
                                    )
                        else:
                            compiler.register_agent_as_toolkit(reg)
            elif mod == "mcp":
                model_id = _extract_model_id(
                    ctx.get("tool_choice_model"), llm_defaults.smart
                )
                feedback_enabled = bool(ctx.get("interaction", False))
                max_turns = int(ctx.get("max_turns") or 5) + 2
                compiler.register_agent_as_toolkit(ctx)
            elif mod == "multi_mcp":
                model_id = _extract_model_id(
                    ctx.get("tool_choice_model"), llm_defaults.smart
                )
                feedback_enabled = bool(ctx.get("interaction", False))
                max_turns = int(ctx.get("max_turns") or 5) + 2
                for m_cfg in ctx.get("configs") or []:
                    if isinstance(m_cfg, dict):
                        compiler.register_agent_as_toolkit(m_cfg, lazy_load=True)

        if generation:
            gen = generation[0]
            if not context:
                model_id = _extract_model_id(gen.get("model"), llm_defaults.default)
            if gen.get("system_prompt"):
                prompt_parts.insert(0, str(gen["system_prompt"]))
            if gen.get("prompt"):
                prompt_parts.append(str(gen["prompt"]))
            if gen.get("citations"):
                enabled_skills.append("citation_grounded_answer")

        harness_cfg: dict[str, Any] = {
            "model": model_id,
            "feedback_enabled": feedback_enabled,
            "enabled_skills": enabled_skills,
        }
        if prompt_parts:
            harness_cfg["system_prompt"] = "\n\n".join(prompt_parts)
        if max_turns is not None:
            harness_cfg["usage_limits"] = {"max_turns": max_turns}

        return {
            "execution_mode": "loop",
            "harness_config": harness_cfg,
            "workflow_code": None,
            "toolkits": compiler.toolkits,
        }

    # Compile multi-stage pipeline to RestrictedPython Code Mode script
    restart_cfg = next(
        (item for item in postprocess if item.get("module") == "restart"), None
    )
    max_attempts = (int(restart_cfg.get("retries") or 2) + 1) if restart_cfg else 1

    lines: list[str] = [
        "queries = [question]",
        "contexts = []",
        "final_answer = ''",
        "final_citations = None",
        "is_answered = True",
        "related_questions = []",
        "metadata = {}",
    ]

    loop_indent = "    " if restart_cfg is not None else ""
    if restart_cfg is not None:
        lines.extend(
            [
                "wf_attempt = 0",
                "wf_answered = False",
                f"while not wf_answered and wf_attempt < {max_attempts}:",
                "    if wf_attempt == 0:",
            ]
        )
        prep_lines = compiler._emit_preprocess_lines(preprocess, "        ")
        lines.extend(prep_lines or ["        pass"])
    else:
        lines.extend(compiler._emit_preprocess_lines(preprocess, ""))

    # Context stage: use parallel([...]) when all context items are simple retrievers
    simple_parallel = len(context) >= 1 and all(
        item.get("module")
        in {
            "basic_ask",
            "ask",
            "sync",
            "google",
            "perplexity",
            "perplexity_search",
            "http",
            "a2a",
        }
        and not item.get("fallback")
        and not item.get("next_agent")
        and not item.get("prune_context", True)
        for item in context
    )
    if simple_parallel:
        call_fns = [
            fn
            for item in context
            if (fn := compiler.register_agent_as_toolkit(item)[1])
        ]
        if len(call_fns) == 1:
            lines.append(
                f"{loop_indent}wf_batches = parallel([call({call_fns[0]}, question=wf_q) for wf_q in queries])"
            )
        elif len(call_fns) > 1:
            calls_tuple = ", ".join(f"call({fn}, question=wf_q)" for fn in call_fns)
            lines.append(
                f"{loop_indent}wf_batches = parallel([wf_c for wf_q in queries for wf_c in ({calls_tuple})])"
            )
        lines.append(f"{loop_indent}for wf_b in wf_batches:")
        lines.append(f"{loop_indent}    contexts.extend(wf_b.get('contexts', []))")
    elif context:
        lines.append(f"{loop_indent}for wf_q in queries:")
        body_lines: list[str] = []
        for item in context:
            body_lines.extend(
                compiler._emit_context_agent_lines(
                    item, "wf_q", "contexts", loop_indent + "    "
                )
            )
        lines.extend(body_lines or [f"{loop_indent}    pass"])

    lines.extend(compiler._emit_generation_lines(generation, loop_indent))

    if restart_cfg is not None:
        eval_ns = compiler._ensure_shared_toolkit(
            "evaluation",
            {
                "toolkit": "evaluation",
                "model": _llm_dict(restart_cfg.get("model"), llm_defaults.default),
            },
        )
        lines.extend(
            [
                f"    wf_gap = {eval_ns}__detect_missing_info(question=question, contexts=contexts)",
                f"    if wf_gap['answered'] or wf_attempt + 1 >= {max_attempts}:",
                "        wf_answered = True",
                "    else:",
                "        queries = wf_gap['questions']",
                "    wf_attempt = wf_attempt + 1",
            ]
        )

    lines.extend(compiler._emit_postprocess_lines(postprocess, ""))
    lines.append(
        "output({"
        "'answer': final_answer, "
        "'citations': final_citations, "
        "'contexts': contexts, "
        "'related_questions': related_questions, "
        "'is_answered': is_answered, "
        "'metadata': metadata"
        "})"
    )

    workflow_code = "\n".join(lines) + "\n"
    compile_restricted(workflow_code, filename="<migrated_workflow>", mode="exec")

    return {
        "execution_mode": "code",
        "harness_config": {
            "model": "",
            "codemode_limits": {"max_nested_calls": 50},
            "usage_limits": {"max_tool_calls": 60},
        },
        "workflow_code": workflow_code,
        "toolkits": compiler.toolkits,
    }


def _decode_json(raw: Any, default: Any) -> Any:
    if raw is None:
        return default
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def migrate_database_workflows_to_harness(
    conn: Connection,
    *,
    external_migrations: Mapping[str, MigrationHook] | None = None,
) -> int:
    """Migrate all stored 4-stage PostgreSQL workflows into unified Harness columns & toolkits."""
    workflows = conn.execute(
        text(
            "SELECT account, agent_id, workflow_id FROM retrieval_agent_workflow "
            "ORDER BY account, agent_id, workflow_id"
        )
    ).fetchall()

    migrated = 0
    stage_tables = (
        ("preprocess", "retrieval_agent_preprocess", "preprocess"),
        ("context", "retrieval_agent_context", "context"),
        ("generation", "retrieval_agent_generation", "generation"),
        ("postprocess", "retrieval_agent_postprocess", "postprocess"),
    )

    for account, agent_id, workflow_id in workflows:
        snapshot: dict[str, list[dict[str, Any]]] = {}
        stage_configs: dict[str, list[dict[str, Any]]] = {}

        for key, table, col in stage_tables:
            rows = conn.execute(
                text(
                    f"SELECT id, {col}, created, modified FROM {table} "
                    "WHERE account = :account AND agent_id = :agent_id "
                    "AND workflow_id = :workflow_id ORDER BY created ASC, id ASC"
                ),
                {
                    "account": account,
                    "agent_id": agent_id,
                    "workflow_id": workflow_id,
                },
            ).fetchall()
            snapshot[key] = [
                {
                    "id": str(r[0]),
                    "config": _decode_json(r[1], {}),
                    "created": (
                        r[2].isoformat() if hasattr(r[2], "isoformat") else r[2]
                    ),
                    "modified": (
                        r[3].isoformat() if hasattr(r[3], "isoformat") else r[3]
                    ),
                }
                for r in rows
            ]
            stage_configs[key] = [
                {"id": str(r[0]), **_decode_json(r[1], {})} for r in rows
            ]

        compiled = transpile_stages_to_harness(
            preprocess=stage_configs["preprocess"],
            context=stage_configs["context"],
            generation=stage_configs["generation"],
            postprocess=stage_configs["postprocess"],
            external_migrations=external_migrations,
        )

        is_pg = conn.dialect.name == "postgresql"
        json_hc = "CAST(:harness_config AS jsonb)" if is_pg else ":harness_config"
        json_snap = "CAST(:snapshot AS jsonb)" if is_pg else ":snapshot"
        uuid_expr = "CAST(:id AS uuid)" if is_pg else ":id"
        json_cfg = "CAST(:config AS jsonb)" if is_pg else ":config"

        conn.execute(
            text(
                "UPDATE retrieval_agent_workflow "
                f"SET execution_mode = :execution_mode, "
                f"    harness_config = {json_hc}, "
                "    workflow_code = :workflow_code, "
                f"    legacy_stages_snapshot = {json_snap} "
                "WHERE account = :account AND agent_id = :agent_id "
                "AND workflow_id = :workflow_id"
            ),
            {
                "execution_mode": compiled["execution_mode"],
                "harness_config": json.dumps(compiled["harness_config"]),
                "workflow_code": compiled["workflow_code"],
                "snapshot": json.dumps(snapshot),
                "account": account,
                "agent_id": agent_id,
                "workflow_id": workflow_id,
            },
        )

        for tk in compiled["toolkits"]:
            tk_copy = dict(tk)
            tk_id = str(tk_copy.pop("id", None) or uuid4())
            namespace = str(tk_copy.pop("namespace"))
            provider = str(tk_copy.pop("toolkit"))
            title = tk_copy.pop("title", None)
            lazy_load = bool(tk_copy.pop("lazy_load", False))
            ingest_skills = bool(tk_copy.pop("ingest_skills", True))
            conn.execute(
                text(
                    "INSERT INTO agent_workflow_toolkits "
                    "(id, account, agent_id, workflow_id, namespace, toolkit, "
                    " title, lazy_load, ingest_skills, config) "
                    f"VALUES ({uuid_expr}, :account, :agent_id, :workflow_id, "
                    "        :namespace, :toolkit, :title, :lazy_load, :ingest_skills, "
                    f"        {json_cfg})"
                ),
                {
                    "id": tk_id,
                    "account": account,
                    "agent_id": agent_id,
                    "workflow_id": workflow_id,
                    "namespace": namespace,
                    "toolkit": provider,
                    "title": title,
                    "lazy_load": lazy_load,
                    "ingest_skills": ingest_skills,
                    "config": json.dumps(tk_copy),
                },
            )

        migrated += 1

    return migrated


def restore_database_workflows_from_snapshot(conn: Connection) -> int:
    """Restore the 4 legacy stage tables from `retrieval_agent_workflow.legacy_stages_snapshot`."""
    is_pg = conn.dialect.name == "postgresql"
    uuid_expr = "CAST(:id AS uuid)" if is_pg else ":id"
    json_cfg = "CAST(:config AS jsonb)" if is_pg else ":config"

    rows = conn.execute(
        text(
            "SELECT account, agent_id, workflow_id, legacy_stages_snapshot "
            "FROM retrieval_agent_workflow WHERE legacy_stages_snapshot IS NOT NULL"
        )
    ).fetchall()

    stage_tables = (
        ("preprocess", "retrieval_agent_preprocess", "preprocess"),
        ("context", "retrieval_agent_context", "context"),
        ("generation", "retrieval_agent_generation", "generation"),
        ("postprocess", "retrieval_agent_postprocess", "postprocess"),
    )
    restored = 0
    for account, agent_id, workflow_id, raw_snapshot in rows:
        snapshot = _decode_json(raw_snapshot, {})
        for key, table, col in stage_tables:
            for item in snapshot.get(key, []):
                conn.execute(
                    text(
                        f"INSERT INTO {table} (id, account, agent_id, workflow_id, {col}) "
                        f"VALUES ({uuid_expr}, :account, :agent_id, :workflow_id, "
                        f"        {json_cfg})"
                    ),
                    {
                        "id": item.get("id") or str(uuid4()),
                        "account": account,
                        "agent_id": agent_id,
                        "workflow_id": workflow_id,
                        "config": json.dumps(item.get("config") or {}),
                    },
                )
        restored += 1
    return restored


__all__ = [
    "MigrationHook",
    "migrate_database_workflows_to_harness",
    "restore_database_workflows_from_snapshot",
    "transpile_stages_to_harness",
]
