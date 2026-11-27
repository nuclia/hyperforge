# Hyperforge Unified Harness Architecture & Migration Plan

## 1. Overview & Motivation

Hyperforge previously evolved two parallel runtime implementations behind `hyperforge.engine`:

1. **Fixed 4-Stage Retrieval Pipeline (`RetrievalAgent` / `ContextAgent`)**:
   - Located in `hyperforge/src/hyperforge/retrieval/` and `hyperforge/src/hyperforge/context/`.
   - Executes a rigid `while memory.restart:` pipeline across `preprocess` $\rightarrow$ `context` $\rightarrow$ `generation` $\rightarrow$ `postprocess` over a mutable `QuestionMemory`.
   - Bakes opinionated LLM calls directly into `ContextAgent` (relevance validation, `block-XX` citation pruning, `fallback` recursion on missing queries, and `next_agent` rephrasing).
   - Hardcoded into the PostgreSQL schema (`retrieval_agent_{preprocess,context,generation,postprocess}`), REST workflow endpoints, and standalone JSON configuration.

2. **Standard Agentic Loop (`AgentHarness` / `harness_sdk`)**:
   - Located in `hyperforge/src/hyperforge/harness_sdk/`, `hyperforge/src/hyperforge/harness/`, `hyperforge/src/hyperforge/procedural/`, and `hyperforge/src/hyperforge/codemode/`.
   - Provides an asynchronous, event-sourced tool loop with Pydantic-validated `HarnessTool`s, `lazy_load` tool discovery (`search_tools`, `activate_tools`, `call_tool`), sub-agents (`spawn_agent`, `send_message`, `wait_agent`), user `feedback`, `UsageLimits`, `DecisionCache`, `ProceduralGuidance`, and a hardened `RestrictedPython` sandbox (`codemode`).

### Goal of the Reimplementation
Unify all of Hyperforge onto **`AgentHarness` as the single runtime engine**:
- Keep the core loop **non-opinionated**, shaping behavior orthogonally via **Skills**, **Prompts**, **Procedural Graphs**, or **Code Mode**.
- Extend **Code Mode** so deterministic multi-step workflows (including all existing `preprocess`, `context`, `generation`, `postprocess`, `fallback`, `next_agent`, `conditional`, and `restart` patterns) can be authored and executed as restricted Python code over the same tools.
- Convert all **non-loop agents** into stateless, Pydantic-typed **`Toolkit`s** (`HarnessTool` providers).
- Decompose **loop agents** (`mcp`, `smart`, `restricted`, `a2a`) to separate their underlying tools/connectors from any agentic loop so their tools can be called directly in both the Harness loop and Code Mode.
- Keep **connector toolkits isolated per package** with standard plugin expansion (`entry_points` + `load_modules`) so external systems (e.g., `progress-agents`) can register custom drivers, toolkits, skills, and migration hooks.
- Ensure **every toolkit can bundle `HarnessSkill`s** that are automatically ingested when the toolkit is attached to a workflow.
- Provide a **lossless PostgreSQL Alembic migration** from the 4-stage schema to the unified Harness schema.

---

## 2. Target Architecture

```
                         ┌─────────────────────────────────────────┐
                         │       HarnessWorkflowConfig             │
                         │  mode: "loop" | "code"                  │
                         │  toolkits, prompts, skills, graph, code │
                         └────────────────────┬────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              AgentHarness Runtime                                   │
│                                                                                     │
│  Composable Guidance (Optional, Non-Opinionated):                                   │
│   ├── Prompts: Rendered PromptConfig / Jinja templates + workflow arguments         │
│   ├── Skills: Toolkit-ingested & custom skills (<available_skills> + `load_skill`)  │
│   └── Procedural Graph: Pre-decision localized graph advisor (`ProceduralGuidance`) │
│                                                                                     │
│  Execution Surfaces:                                                                │
│   ├── Native Tool Calls ──► Concurrent `HarnessTool.execute()` (eager or lazy_load) │
│   ├── Sub-Agents ─────────► `spawn_agent` / `wait_agent` / `create_subagent_tool`   │
│   └── Extended Code Mode ─► Hardened Sandbox (`SandboxRunner`)                      │
│                              ├── Active `HarnessTool`s (NucliaDB, MCP, Web, HTTP…)  │
│                              ├── `call(...)` + `parallel([...])` batch dispatch     │
│                              └── Pipeline toolkits (rephrase, validate_contexts,    │
│                                  synthesize_answer, evaluate_remi, check_condition) │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Non-Opinionated Loop, Skills, Prompts, and Procedural Graphs
1. **Unopinionated Default Loop**: `AgentHarness` runs a standard streaming tool loop (`ModelClient.stream` $\rightarrow$ concurrent `HarnessTool` calls $\rightarrow$ final answer) without mandatory planning stages.
2. **Skills (`hyperforge.harness_sdk.skills`)**:
   - `HarnessSkill` defines `id`, `name`, `description`, `instructions`, `recommended_tools`, `code_examples`, and `preload`.
   - `preload=True` injects compact connector rules directly into the system prompt.
   - `preload=False` (default) lists only `id`, `name`, and `description` under `<available_skills>` in the system prompt; calling the core `load_skill` tool loads the full instructions, activates any `lazy_load` tools in `recommended_tools`, and persists a `SKILLS_LOADED` event for conversation reload.
3. **Prompts (`hyperforge.prompts`)**:
   - Stored `PromptConfig` templates render system/user instructions with workflow `arguments` and can also be exposed as skills or MCP prompts.
4. **Procedural Graphs (`hyperforge.procedural`)**:
   - Opt-in `ProceduralGuidanceConfig` + `ProceduralGraph` (`STATUS`, `ACTION`, `REASONING`) advises the solver before each decision and supports offline `evolve()`.

---

## 3. Extensible Per-Connector Toolkits & Automatic Skill Ingestion

### 3.1 Connector Isolation & External Plugin Discovery
- Each external data source or integration remains an isolated package (`hyperforge_nucliadb`, `hyperforge_mcp`, `hyperforge_a2a`, `hyperforge_google`, `hyperforge_perplexity`, `hyperforge_http`, `hyperforge_external`).
- Generic pipeline operations live in core `hyperforge.toolkits` (`rephrase`, `context_ops`, `synthesis`, `decision`, `evaluation`).
- **External Expansion (`configure.py`)**:
  - Supports both explicit `load_modules` scanning (`@driver`, `@toolkit`, `@skill`) and standard Python packaging `importlib.metadata.entry_points`:
    - `hyperforge.drivers`
    - `hyperforge.toolkits`
    - `hyperforge.skills`
    - `hyperforge.migrations`
  - External packages (such as `progress-agents` connectors for `snowflake`, `salesforce`, `sql`, `cypher`, `sparql`, `marklogic`, `sitefinity`, `brave`, `pandas`) register their own drivers, toolkits, bundled skills, and legacy DB migration rules without modifying `hyperforge` core.

### 3.2 Toolkit Contract & Skill Bundling
Every `BaseToolkit` provides:
- `preload(runtime: ToolkitRuntime) -> None`: Async initialization (e.g., MCP tool/prompt discovery).
- `build_tools() -> Sequence[HarnessTool]`: Tools automatically namespaced as `{namespace}__{tool_name}` with optional `lazy_load` and `ResultPayloadBudget`.
- `build_skills() -> Sequence[HarnessSkill]`: Connector playbooks automatically scoped to `{namespace}` (interpolating `{namespace}` in instructions/examples and prefixing `recommended_tools`) and ingested into `AgentHarness` whenever `ingest_skills=True`.

---

## 4. Extending Code Mode for Deterministic & Hybrid Pipeline Workflows

To express all legacy 4-stage workflows in Code Mode without sacrificing concurrency, telemetry, or structured citations:

1. **Concurrent Fan-Out (`call` + `parallel`)**:
   - RestrictedPython worker code is synchronous; add `call(fn, *args, **kwargs)` and `parallel([call(...), ...])` to worker globals.
   - Host dispatcher executes batched calls concurrently via `asyncio.gather` while enforcing `max_nested_calls` and `UsageLimits.max_tool_calls` and emitting nested `TOOL_REQUESTED`/`TOOL_COMPLETED` events.
2. **Read-Only Workflow Context Globals**:
   - Inject sanitized `arguments`, `history`, `rules`, and `user_info` alongside `question`.
3. **Step Telemetry & Structured `WorkflowOutput`**:
   - Add `emit_step(title, reason="", value="")` host capability.
   - Support structured workflow outputs (`answer`, `citations`, `contexts`, `related_questions`, `visualizations`, `is_answered`) in `output(...)`.
4. **Direct Code Workflow Mode (`AgentHarness.run_code_workflow`)**:
   - Executes `workflow_code` directly inside an `AgentHarness` turn (`TURN_STARTED` $\rightarrow$ `codemode` $\rightarrow$ `TURN_COMPLETED`) with zero outer orchestrator LLM calls when `execution_mode == "code"`.

---

## 5. Agent-to-Toolkit Mapping & Loop-Agent Decomposition

### 5.1 Non-Loop Agents $\rightarrow$ Pure `Toolkit`s
| Legacy Package | Replacement Toolkit / Tools | Bundled Skill Focus |
| :--- | :--- | :--- |
| `hyperforge_nucliadb` (`basic_ask`, `advanced_ask`, `ask`, `sync`) | `NucliaDBToolkit` & `NucliaDBSyncToolkit` (`ask`, `advanced_ask`, `find`, `search_by_title`, `ask_by_title`, `catalog_search`, `facets_count`, `facets_search`, `list_labels`, `get_images`, `analyze_query`, `rerank`, `choose_source`) | Filter expression syntax (`/l/{labelset}/{label}`), title vs semantic search selection, facet counting pitfalls |
| `hyperforge_google` | `GoogleSearchToolkit` (`google_search`) | Grounded web search query formulation & URL attribution |
| `hyperforge_perplexity` + `perplexity_search` | `PerplexityToolkit` (`perplexity_answer`, `perplexity_search`) | Choosing raw snippet search vs grounded synthesis |
| `hyperforge_http` | `HTTPToolkit` (`http_fetch`) | Safe endpoint parameter usage |
| `hyperforge_external` | `WebhookToolkit` (`webhook_call`) | Payload construction & error handling |
| `hyperforge_rephrase` | Core `RephraseToolkit` (`rephrase_query`, `split_query`) | Sub-question decomposition guidelines |
| `context/agent.py` helpers | Core `ContextOpsToolkit` (`validate_contexts`, `rephrase_for_next`) | Context pruning & multi-hop chaining |
| `hyperforge_summarize` + `generate` + `passthrough` | Core `SynthesisToolkit` (`synthesize_answer`, `generate_text`, `passthrough`) | Two-level markdown footnote citation rules |
| `hyperforge_conditional` | Core `DecisionToolkit` (`check_condition`, `choose_option`, `extract_structured`, `transform`) | Structured classification & extraction |
| `hyperforge_remi` + `related` + `restart` | Core `EvaluationToolkit` (`evaluate_remi`, `suggest_related_questions`, `detect_missing_info`) | Groundedness verification & iterative gap recovery |
| `hyperforge_static` + `static_string` + `historical` | `StaticContextToolkit` / `PromptConfig` + `HarnessStorageProtocol.recall` | — |

### 5.2 Loop Agents $\rightarrow$ Separated Tools + Reusable Harness Primitives
1. **`hyperforge_mcp` (`MCPAgent`, `MultiMCPAgent`)**:
   - **`MCPToolkit` (No LLM Loop)**: Discovers remote MCP tools via `preload()`, normalizes schemas, enforces `ResultPayloadBudget`, exposes native `HarnessTool`s (with `lazy_load=True` replacing `MultiMCPAgent` for large servers), and ingests remote MCP server instructions & prompts as `HarnessSkill`s. Every MCP tool becomes directly callable in Code Mode (`server__tool_name(...)`).
   - Delegated multi-turn MCP reasoning, when desired, uses `create_subagent_tool(tools=mcp_toolkit.get_tools(), skills=mcp_toolkit.get_skills())`.
2. **`hyperforge_smart` (`SmartAgent`)**:
   - Child agents register directly as workflow toolkits.
   - `ResultPayloadBudget` moves onto `HarnessTool`; duplicate failed/empty call protection moves into `AgentHarness`.
   - `reactive` mode is the native `AgentHarness` loop; `plan_execute` becomes the built-in `plan_and_execute` `HarnessSkill` (plus optional `create_subagent_tool` for calling a scoped agent loop from Code Mode).
3. **`hyperforge_restricted` (`PythonAgent`)**:
   - Controller helpers (`choose`, `extract`, `transform`) move to `DecisionToolkit`; execution moves to hardened `codemode`.
4. **`hyperforge_a2a` (`A2AClientAgent`)**:
   - `A2AToolkit` exposes `a2a_send` as a `HarnessTool`, bridging remote `input_required` feedback via `harness.request_feedback()`.

---

## 6. Database Schema & Lossless Alembic Migration

### 6.1 Schema Changes (Revising Head `8c7d2e4f91ab`)
- **Preserved Tables**: `retrieval_agent_config`, `retrieval_agents_drivers`, `retrieval_agent_prompts`, `sync_oauth_credentials`, `download_requests`.
- **Updated `retrieval_agent_workflow`**:
  - `execution_mode` (`VARCHAR`, `'loop' | 'code'`, default `'loop'`)
  - `harness_config` (`JSONB`, non-null)
  - `workflow_code` (`TEXT`, nullable)
  - `legacy_stages_snapshot` (`JSONB`, nullable — full pre-migration snapshot for lossless `downgrade()`)
- **New `agent_workflow_toolkits`**:
  - `(id, account, agent_id, workflow_id, namespace, toolkit, title, lazy_load, ingest_skills, config, created, modified)`
- **New `agent_skills`**:
  - `(id, account, agent_id, skill_id, name, description, instructions, recommended_tools, code_examples, preload, meta, created, modified)`
- **New `HarnessStorageProtocol` Tables**:
  - `harness_conversations`, `harness_events`, `harness_memories`

### 6.2 Data Migration Logic
1. Snapshot all rows from `retrieval_agent_{preprocess,context,generation,postprocess}` into `retrieval_agent_workflow.legacy_stages_snapshot`.
2. Pure `smart` + `summarize` workflows convert to `execution_mode = 'loop'` with child agents extracted into `agent_workflow_toolkits` (`ingest_skills = True`) and `plan_and_execute` / `citation_grounded_answer` skills enabled as appropriate.
3. Multi-stage, conditional, fallback, `next_agent`, `restart`, and `restricted` workflows convert to `execution_mode = 'code'`, extracting all referenced agents (including external plugins via `hyperforge.migrations` entry points or generic pass-through mapping) into `agent_workflow_toolkits` and compiling the stage graph into RestrictedPython-validated `workflow_code`.
4. Drop the 4 legacy stage tables (`downgrade()` recreates them from `legacy_stages_snapshot`).

---

## 7. Implementation Phases

- **Phase 1 — Core `harness_sdk` Foundation**:
  1. `HarnessSkill`, `<available_skills>` prompt injection, `load_skill` core tool, and `SKILLS_LOADED` persistence/replay.
  2. `BaseToolkit`, `ToolkitRuntime`, and automatic namespaced skill ingestion.
  3. `@toolkit`, `@skill`, and `importlib.metadata.entry_points` plugin discovery in `configure.py`.
  4. `ResultPayloadBudget` on `HarnessTool`, `create_subagent_tool()`, `PromptConfig` rendering, `PostgresHarnessStorage`, and `HarnessEventBridge` (`HarnessEvent` $\rightarrow$ `AragAnswer` / `Step` / `AnswerCitations` / OAuth).
- **Phase 2 — Code Mode Workflow Extensions**:
  1. `call(...)` + `parallel([...])` concurrent batch dispatch in `codemode`.
  2. Workflow context globals (`arguments`, `history`, `rules`, `user_info`), `emit_step`, structured `WorkflowOutput`, and `AgentHarness.run_code_workflow()`.
- **Phase 3 — Core Pipeline Toolkits & Per-Connector Toolkits (with Bundled Skills)**:
  1. Core toolkits (`rephrase`, `context_ops`, `synthesis`, `decision`, `evaluation`) + built-in skills.
  2. Per-connector toolkits (`nucliadb`, `mcp`, `a2a`, `google`, `perplexity`, `http`, `external`) with connector skills; decompose `MCPAgent`, `SmartAgent`, and `PythonAgent`.
- **Phase 4 — Alembic Migration, Engine/API Cutover & Legacy Removal**:
  1. Workflow transpiler + Alembic migration + roundtrip tests.
  2. Cut over `db/agents.py`, `standalone/`, `api/v1/`, `server/session.py`, and `engine.py` exclusively to `AgentHarness`; remove `hyperforge/retrieval/` and `hyperforge/context/`.
