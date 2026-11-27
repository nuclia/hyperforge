import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, cast

from nuclia.lib.nua import AsyncNuaClient

from hyperforge.configure import (
    GLOBAL_REGISTRY,
    load_all_configurations,
    load_entry_point_plugins,
)
from hyperforge.db.workflow_migration import transpile_stages_to_harness
from hyperforge.harness import HarnessAgent, HarnessAgentConfig
from hyperforge.harness_sdk import HarnessStorageProtocol
from hyperforge.interaction import AragAnswer
from hyperforge.llm import (
    AsyncLocalOpenAIClient,
    NoopNuaClient,
    NuaBaseModel,
    NUAConnection,
)
from hyperforge.manager import Manager
from hyperforge.memory.memory import BaseSessionMemory, QuestionMemory, SessionMemory
from hyperforge.models import HistoryQuestionAnswer
from hyperforge.retrieval.config import RetrievalAgentConfig

logger = logging.getLogger(__name__)


@dataclass
class State:
    agent: Any
    manager: Manager


def ensure_harness_config(
    config: Dict[str, Any] | RetrievalAgentConfig | HarnessAgentConfig,
) -> HarnessAgentConfig:
    """Resolve any workflow config (native Harness or legacy 4-stage) into `HarnessAgentConfig`."""
    if isinstance(config, HarnessAgentConfig):
        return config
    raw_dict: dict[str, Any]
    if isinstance(config, RetrievalAgentConfig):
        raw_dict = {
            "drivers": [d.model_dump() for d in config.drivers],
            "rules": config.rules.model_dump(),
            "memory": config.memory.model_dump(),
            "preprocess": [x.model_dump() for x in config.preprocess],
            "context": [x.model_dump() for x in config.context],
            "generation": [x.model_dump() for x in config.generation],
            "postprocess": [x.model_dump() for x in config.postprocess],
        }
    else:
        raw_dict = dict(config)

    if any(
        key in raw_dict
        for key in (
            "agents",
            "toolkits",
            "skills",
            "execution_mode",
            "workflow_code",
        )
    ):
        return HarnessAgentConfig.model_validate(raw_dict)

    ext_migrations = {
        k: v.handler
        for k, v in GLOBAL_REGISTRY.migrations.items()
        if v.handler is not None
    }
    compiled = transpile_stages_to_harness(
        preprocess=list(raw_dict.get("preprocess") or []),
        context=list(raw_dict.get("context") or []),
        generation=list(raw_dict.get("generation") or []),
        postprocess=list(raw_dict.get("postprocess") or []),
        external_migrations=ext_migrations,
    )
    harness_settings = dict(compiled["harness_config"])
    enabled_skill_ids = harness_settings.pop("enabled_skills", []) or []
    skills = [
        reg.skill
        for sid in enabled_skill_ids
        if (reg := GLOBAL_REGISTRY.skills.get(sid)) and reg.skill is not None
    ]
    return HarnessAgentConfig.model_validate(
        {
            "execution_mode": compiled["execution_mode"],
            "workflow_code": compiled["workflow_code"],
            "toolkits": compiled["toolkits"],
            "drivers": raw_dict.get("drivers", []),
            "skills": skills,
            "rules": raw_dict.get("rules", {}),
            "memory": raw_dict.get("memory", {}),
            **harness_settings,
        }
    )


async def init(
    config: Optional[Dict[str, Any]] = None,
    agent_id: str = "default",
    internal_nua: bool = False,
    internal_nua_api: str = "http://predict.learning.svc.cluster.local:8080",
    local_openai: Optional[str] = None,
    local_openai_model: Optional[str] = None,
    external_nua_api_key: Optional[str] = None,
    loaded_modules: list[str] = [],
    retrieval_config: Optional[RetrievalAgentConfig | HarnessAgentConfig] = None,
    session_id: str = "default_session",
    memory_klass: type[BaseSessionMemory] = SessionMemory,
    storage: Optional[HarnessStorageProtocol] = None,
) -> Tuple[State, SessionMemory]:
    from hyperforge.configure import scan

    for load_module in loaded_modules:
        try:
            scan(load_module)
            load_all_configurations(load_module)
        except ImportError:
            logger.error(f"Module {load_module} could not be loaded")
    load_entry_point_plugins()

    if retrieval_config is None:
        if config is None:
            raise ValueError("Either config or retrieval_config must be provided")
        resolved_config = ensure_harness_config(config)
    else:
        resolved_config = ensure_harness_config(retrieval_config)

    state = await get_state(
        agent_id=agent_id,
        config=resolved_config,
        internal_nua=internal_nua,
        internal_nua_api=internal_nua_api,
        local_openai=local_openai,
        local_openai_model=local_openai_model,
        external_nua_api_key=external_nua_api_key,
        storage=storage,
    )
    session_memory = memory_klass.from_config(
        resolved_config.memory,
        agent_id=agent_id,
        workflow_id="default",
        rules=resolved_config.rules,
    )
    session_memory.init(session_id)
    return state, session_memory


async def main(
    config: Optional[Dict[str, Any]] = None,
    agent_id: str = "default",
    internal_nua: bool = False,
    internal_nua_api: str = "http://predict.learning.svc.cluster.local:8080",
    local_openai: Optional[str] = None,
    local_openai_model: Optional[str] = None,
    external_nua_api_key: Optional[str] = None,
    question: str = "",
    loaded_modules: list[str] = [],
    retrieval_config: Optional[RetrievalAgentConfig | HarnessAgentConfig] = None,
    callback: Optional[Callable[[AragAnswer], Awaitable[None]]] = None,
    session_id: str = "default_session",
    user_metadata: Optional[Dict[str, str]] = None,
    headers: Optional[Dict[str, str]] = None,
    memory_klass: type[BaseSessionMemory] = SessionMemory,
    streaming: bool = False,
    chat_history: Optional[List[HistoryQuestionAnswer]] = None,
    storage: Optional[HarnessStorageProtocol] = None,
) -> QuestionMemory:
    state: State | None = None
    try:
        state, session_memory = await init(
            config=config,
            agent_id=agent_id,
            internal_nua=internal_nua,
            internal_nua_api=internal_nua_api,
            local_openai=local_openai,
            local_openai_model=local_openai_model,
            external_nua_api_key=external_nua_api_key,
            loaded_modules=loaded_modules,
            retrieval_config=retrieval_config,
            session_id=session_id,
            memory_klass=memory_klass,
            storage=storage,
        )
        question_memory = session_memory.start_question(
            question, streaming=streaming, chat_history=chat_history
        )
        if callback is not None:
            question_memory.set_callback_fn(callback)

        if user_metadata:
            question_memory.session.user_info.update(user_metadata)

        if headers:
            question_memory.headers.update(headers)

        if state.agent is None:
            raise ValueError("Agent could not be initialized")

        await state.agent(
            question_memory,
            state.manager,
        )
    except Exception as e:
        raise e
    finally:
        if state is not None and state.manager is not None:
            await state.manager.aclose()
        GLOBAL_REGISTRY.clear()
    return question_memory


async def engine(
    manager: Manager,
    agent: Any,
    question_memory: QuestionMemory,
    user_metadata: Optional[Dict[str, str]] = None,
) -> None:
    if user_metadata is not None:
        question_memory.session.user_info.update(user_metadata)

    await agent(
        question_memory,
        manager,
    )


async def get_state(
    agent_id: str,
    config: RetrievalAgentConfig | HarnessAgentConfig,
    internal_nua_api: str = "http://predict.learning.svc.cluster.local:8080",
    internal_nua: bool = False,
    local_openai: Optional[str] = None,
    local_openai_model: Optional[str] = None,
    external_nua_api_key: Optional[str] = None,
    account: Optional[str] = None,
    kbid: Optional[str] = None,
    local_openai_model_klass: Optional[type[NuaBaseModel]] = None,
    allow_private_network_endpoints: bool = False,
    storage: Optional[HarnessStorageProtocol] = None,
) -> State:
    resolved_config = ensure_harness_config(config)
    nua: AsyncNuaClient
    if internal_nua:
        nua = cast(
            AsyncNuaClient,
            await NUAConnection.connect_internal(
                kbid=kbid, account=account, url=internal_nua_api
            ),
        )

    elif local_openai is not None:
        if local_openai_model_klass is not None:
            nua = await local_openai_model_klass.model_validate(
                {
                    "key": external_nua_api_key,
                    "local_openai": local_openai,
                    "local_openai_model": local_openai_model,
                }
            ).connect()
        else:
            nua = cast(
                AsyncNuaClient,
                AsyncLocalOpenAIClient(
                    base_url=local_openai,
                    api_key=external_nua_api_key,
                    model=local_openai_model,
                ),
            )

    elif external_nua_api_key is not None:
        nua = await NUAConnection.model_validate(
            {
                "key": external_nua_api_key,
            }
        ).connect()

    else:
        logger.warning(
            "No LLM backend configured — use a no-op client. Toolkits that don't"
            " require LLM calls (e.g. ``static``) will work fine; any tool that"
            " calls NUA will raise a clear error."
        )
        nua = NoopNuaClient()

    manager = await Manager.from_config(
        drivers=resolved_config.drivers,
        nua=nua,
        allow_private_network_endpoints=allow_private_network_endpoints,
    )
    agent = await HarnessAgent.from_config_class(resolved_config, storage=storage)
    return State(manager=manager, agent=agent)
