import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, cast

from nuclia.lib.nua import AsyncNuaClient

from hyperforge.configure import GLOBAL_REGISTRY, load_all_configurations
from hyperforge.interaction import AragAnswer
from hyperforge.llm import NoopNuaClient, NuaBaseModel, NUAConnection
from hyperforge.manager import Manager
from hyperforge.memory.memory import BaseSessionMemory, QuestionMemory, SessionMemory
from hyperforge.models import HistoryQuestionAnswer
from hyperforge.retrieval.agent import RetrievalAgent
from hyperforge.retrieval.config import RetrievalAgentConfig

logger = logging.getLogger(__name__)


@dataclass
class State:
    agent: RetrievalAgent
    manager: Manager


async def init(
    config: Optional[Dict[str, Any]] = None,
    agent_id: str = "default",
    internal_nua: bool = False,
    internal_nua_api: str = "http://predict.learning.svc.cluster.local:8080",
    local_openai: Optional[str] = None,
    local_openai_model: Optional[str] = None,
    external_nua_api_key: Optional[str] = None,
    loaded_modules: list[str] = [],
    retrieval_config: Optional[RetrievalAgentConfig] = None,
    session_id: str = "default_session",
    memory_klass: type[BaseSessionMemory] = SessionMemory,
) -> Tuple[State, SessionMemory]:
    from hyperforge.configure import scan

    for load_module in loaded_modules:
        try:
            scan(load_module)
            load_all_configurations(load_module)
        except ImportError:
            logger.error(f"Module {load_module} could not be loaded")

    if retrieval_config is None:
        if config is None:
            raise ValueError("Either config or retrieval_config must be provided")
        retrieval_config = RetrievalAgentConfig.model_validate(config)

    state = await get_state(
        agent_id=agent_id,
        config=retrieval_config,
        internal_nua=internal_nua,
        internal_nua_api=internal_nua_api,
        local_openai=local_openai,
        local_openai_model=local_openai_model,
        external_nua_api_key=external_nua_api_key,
    )
    session_memory = memory_klass.from_config(
        retrieval_config.memory,
        agent_id=agent_id,
        workflow_id="default",
        rules=retrieval_config.rules,
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
    retrieval_config: Optional[RetrievalAgentConfig] = None,
    callback: Optional[Callable[[AragAnswer], Awaitable[None]]] = None,
    session_id: str = "default_session",
    user_metadata: Optional[Dict[str, str]] = None,
    headers: Optional[Dict[str, str]] = None,
    memory_klass: type[BaseSessionMemory] = SessionMemory,
    streaming: bool = False,
    chat_history: Optional[List[HistoryQuestionAnswer]] = None,
) -> QuestionMemory:
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
        GLOBAL_REGISTRY.clear()
    return question_memory


async def engine(
    manager: Manager,
    agent: RetrievalAgent,
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
    config: RetrievalAgentConfig,
    internal_nua_api: str = "http://predict.learning.svc.cluster.local:8080",
    internal_nua: bool = False,
    local_openai: Optional[str] = None,
    local_openai_model: Optional[str] = None,
    external_nua_api_key: Optional[str] = None,
    account: Optional[str] = None,
    kbid: Optional[str] = None,
    local_openai_model_klass: Optional[type[NuaBaseModel]] = None,
    allow_private_network_endpoints: bool = False,
) -> State:
    nua: AsyncNuaClient
    if internal_nua:
        nua = cast(
            AsyncNuaClient,
            await NUAConnection.connect_internal(
                kbid=kbid, account=account, url=internal_nua_api
            ),
        )

    elif local_openai is not None and local_openai_model_klass is not None:
        nua = await local_openai_model_klass.model_validate(
            {
                "key": external_nua_api_key,
                "local_openai": local_openai,
                "local_openai_model": local_openai_model,
            }
        ).connect()

    elif external_nua_api_key is not None:
        nua = await NUAConnection.model_validate(
            {
                "key": external_nua_api_key,
            }
        ).connect()

    else:
        logger.warning(
            "No LLM backend configured — use a no-op client.  Agents that don't"
            " require LLM calls (e.g. the built-in ``static`` context agent) will"
            " work fine; any agent that actually calls NUA will raise a clear error."
        )
        nua = NoopNuaClient()

    manager = await Manager.from_config(
        drivers=config.drivers,
        nua=nua,
        allow_private_network_endpoints=allow_private_network_endpoints,
    )
    agent = await RetrievalAgent.from_config_class(config)

    return State(manager=manager, agent=agent)
