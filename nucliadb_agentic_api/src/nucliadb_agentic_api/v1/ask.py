import asyncio
import json
from typing import AsyncGenerator, AsyncIterable, Awaitable, Callable, Dict
from uuid import UUID, uuid4

from fastapi import Header, Request, Response
from hyperforge.driver import Driver
from hyperforge.engine import State
from hyperforge.interaction import AnswerOperation, AragAnswer
from hyperforge.manager import Manager
from hyperforge.memory.memory import EphemeralSessionMemory
from hyperforge.nua import AsyncInternalNuaClient
from hyperforge.retrieval.agent import RetrievalAgent
from hyperforge.retrieval.config import RetrievalAgentConfig
from nucliadb_agentic_api.ask.search.metrics import AskMetrics
from nucliadb_models.configuration import AskConfig
from nucliadb_models.resource import NucliaDBRoles
from nucliadb_models.search import (
    AskRequest,
    AugmentedContext,
    KnowledgeboxFindResults,
    NucliaDBClientType,
    PromptContext,
    PromptContextOrder,
    SyncAskResponse,
    parse_max_tokens,
)
from nucliadb_models.security import RequestSecurity
from nucliadb_sdk.v2.exceptions import PreconditionFailed, UnprocessableEntity
from nucliadb_utils.authentication import NucliaUser, requires
from pydantic import ValidationError
from starlette.responses import StreamingResponse

from nucliadb_agentic_api.agentic.transform import transform_agentic_config
from nucliadb_agentic_api.app import HTTPApplication
from nucliadb_agentic_api.ask.exceptions import (
    AnswerJsonSchemaTooLong,
)
from nucliadb_agentic_api.ask.search import rpc
from nucliadb_agentic_api.ask.search.ask import (
    AskResult,
    ask,
    handled_ask_exceptions,
)
from nucliadb_agentic_api.ask.utils.responses import (
    HTTPClientError,
)
from nucliadb_agentic_api.models import AgenticConfigSchema
from nuclia_models.predict.generative_responses import (
    CitationsGenerativeResponse,
    GenerativeChunk,
    ReasoningGenerativeResponse,
    TextGenerativeResponse,
)
from nucliadb_agentic_api.v1.router import router


@router.post(
    "/api/v1/kb/{kbid}/ask",
    status_code=200,
    summary="Ask Knowledge Box",
    description="Ask questions on a Knowledge Box",
    tags=["Search"],
    response_model=SyncAskResponse,
)
@requires(NucliaDBRoles.READER)
async def ask_knowledgebox_endpoint(
    request: Request,
    kbid: str,
    item: AskRequest,
    x_nucliadb_account: str = Header(default="", include_in_schema=False),
    x_ndb_client: NucliaDBClientType = Header(NucliaDBClientType.API),
    x_show_consumption: bool = Header(default=False),
    x_nucliadb_user: str = Header(""),
    x_forwarded_for: str = Header(""),
    x_synchronous: bool = Header(
        default=False,
        description="When set to true, outputs response as JSON in a non-streaming way. "
        "This is slower and requires waiting for entire answer to be ready.",
    ),
) -> StreamingResponse | HTTPClientError | Response:
    current_user: NucliaUser = request.user
    # If present, security groups from AuthorizationBackend overrides any
    # security group of the payload
    if current_user.security_groups:
        if item.security is None:
            item.security = RequestSecurity(groups=current_user.security_groups)
        else:
            item.security.groups = current_user.security_groups

    if item.search_configuration is not None:
        search_config = await rpc.get_search_configuration(
            rpc.get_sdk("reader"), kbid, name=item.search_configuration
        )
        if search_config is None:
            return HTTPClientError(
                status_code=400, detail="Search configuration not found"
            )

        if not isinstance(search_config.config, AskConfig):
            return HTTPClientError(
                status_code=400,
                detail="This search configuration is not valid for `ask`",
            )

        try:
            item = AskRequest.model_validate(
                search_config.config.model_dump(exclude_unset=True)
                | item.model_dump(exclude_unset=True)
            )
        except ValidationError as e:
            detail = json.loads(e.json())
            return HTTPClientError(status_code=422, detail=detail)

    if item.agentic_config_id is not None:
        app: HTTPApplication = request.app
        config = await app.agent_manager.get_agentic_config(
            account=x_nucliadb_account, kbid=kbid, agentic_id=item.agentic_config_id
        )  # raises if not found
        return await create_agentic_response(
            kbid=kbid,
            agentic_config=config,
            ask_request=item,
            user_id=x_nucliadb_user,
            account=x_nucliadb_account,
            internal_nua_api=app.settings.internal_nua_api,
            global_drivers=app.hyperforge_drivers,
            client_type=x_ndb_client,
            origin=x_forwarded_for,
            x_synchronous=x_synchronous,
            extra_predict_headers={
                "X-Show-Consumption": str(x_show_consumption).lower()
            },
        )

    return await create_ask_response(
        kbid=kbid,
        ask_request=item,
        user_id=x_nucliadb_user,
        client_type=x_ndb_client,
        origin=x_forwarded_for,
        x_synchronous=x_synchronous,
        extra_predict_headers={"X-Show-Consumption": str(x_show_consumption).lower()},
    )


@router.post(
    "/api/v1/kb/{kbid}/resource/{rid}/ask",
    status_code=200,
    summary="Ask a resource (by id)",
    description="Ask questions to a resource",
    tags=["Search"],
    response_model=SyncAskResponse,
)
@requires(NucliaDBRoles.READER)
async def resource_ask_endpoint_by_uuid(
    request: Request,
    kbid: str,
    rid: UUID,
    item: AskRequest,
    x_show_consumption: bool = Header(default=False),
    x_nucliadb_account: str = Header(default="", include_in_schema=False),
    x_ndb_client: NucliaDBClientType = Header(NucliaDBClientType.API),
    x_nucliadb_user: str = Header(""),
    x_forwarded_for: str = Header(""),
    x_synchronous: bool = Header(
        False,
        description="When set to true, outputs response as JSON in a non-streaming way. "
        "This is slower and requires waiting for entire answer to be ready.",
    ),
) -> StreamingResponse | HTTPClientError | Response:
    current_user: NucliaUser = request.user
    # If present, security groups from AuthorizationBackend overrides any
    # security group of the payload
    if current_user.security_groups:
        if item.security is None:
            item.security = RequestSecurity(groups=current_user.security_groups)
        else:
            item.security.groups = current_user.security_groups

    if item.agentic_config_id is not None:
        app: HTTPApplication = request.app
        config = await app.agent_manager.get_agentic_config(
            account=x_nucliadb_account, kbid=kbid, agentic_id=item.agentic_config_id
        )  # raises if not found
        return await create_agentic_response(
            kbid=kbid,
            agentic_config=config,
            account=x_nucliadb_account,
            internal_nua_api=app.settings.internal_nua_api,
            global_drivers=app.hyperforge_drivers,
            ask_request=item,
            user_id=x_nucliadb_user,
            client_type=x_ndb_client,
            origin=x_forwarded_for,
            x_synchronous=x_synchronous,
            extra_predict_headers={
                "X-Show-Consumption": str(x_show_consumption).lower()
            },
        )

    return await create_ask_response(
        kbid=kbid,
        ask_request=item,
        user_id=x_nucliadb_user,
        client_type=x_ndb_client,
        origin=x_forwarded_for,
        x_synchronous=x_synchronous,
        resource=str(rid),
        extra_predict_headers={"X-Show-Consumption": str(x_show_consumption).lower()},
    )


@router.post(
    "/api/v1/kb/{kbid}/slug/{slug}/ask",
    status_code=200,
    summary="Ask a resource (by slug)",
    description="Ask questions to a resource",
    tags=["Search"],
    response_model=SyncAskResponse,
)
@requires(NucliaDBRoles.READER)
async def resource_ask_endpoint_by_slug(
    request: Request,
    kbid: str,
    slug: str,
    item: AskRequest,
    x_show_consumption: bool = Header(default=False),
    x_ndb_client: NucliaDBClientType = Header(NucliaDBClientType.API),
    x_nucliadb_user: str = Header(""),
    x_nucliadb_account: str = Header(default="", include_in_schema=False),
    x_forwarded_for: str = Header(""),
    x_synchronous: bool = Header(
        False,
        description="When set to true, outputs response as JSON in a non-streaming way. "
        "This is slower and requires waiting for entire answer to be ready.",
    ),
) -> StreamingResponse | HTTPClientError | Response:
    resource_id = await rpc.get_resource_uuid_from_slug(
        rpc.get_sdk("reader"), kbid, slug
    )
    if resource_id is None:
        return HTTPClientError(status_code=404, detail="Resource not found")

    current_user: NucliaUser = request.user
    # If present, security groups from AuthorizationBackend overrides any
    # security group of the payload
    if current_user.security_groups:
        if item.security is None:
            item.security = RequestSecurity(groups=current_user.security_groups)
        else:
            item.security.groups = current_user.security_groups

    if item.agentic_config_id is not None:
        app: HTTPApplication = request.app
        config = await app.agent_manager.get_agentic_config(
            account=x_nucliadb_account, kbid=kbid, agentic_id=item.agentic_config_id
        )  # raises if not found
        return await create_agentic_response(
            kbid=kbid,
            account=x_nucliadb_account,
            internal_nua_api=app.settings.internal_nua_api,
            agentic_config=config,
            ask_request=item,
            user_id=x_nucliadb_user,
            client_type=x_ndb_client,
            origin=x_forwarded_for,
            x_synchronous=x_synchronous,
            global_drivers=app.hyperforge_drivers,
            resource=str(resource_id),
            extra_predict_headers={
                "X-Show-Consumption": str(x_show_consumption).lower()
            },
        )

    return await create_ask_response(
        kbid=kbid,
        ask_request=item,
        user_id=x_nucliadb_user,
        client_type=x_ndb_client,
        origin=x_forwarded_for,
        x_synchronous=x_synchronous,
        resource=resource_id,
        extra_predict_headers={"X-Show-Consumption": str(x_show_consumption).lower()},
    )


@handled_ask_exceptions
async def create_ask_response(
    kbid: str,
    ask_request: AskRequest,
    user_id: str,
    client_type: NucliaDBClientType,
    origin: str,
    x_synchronous: bool,
    resource: str | None = None,
    extra_predict_headers: dict[str, str] | None = None,
) -> Response:
    ask_request.max_tokens = parse_max_tokens(ask_request.max_tokens)
    try:
        ask_result: AskResult = await ask(
            search_sdk=rpc.get_sdk("search"),
            reader_sdk=rpc.get_sdk("reader"),
            kbid=kbid,
            ask_request=ask_request,
            user_id=user_id,
            client_type=client_type,
            origin=origin,
            resource=resource,
            extra_predict_headers=extra_predict_headers,
        )

    except AnswerJsonSchemaTooLong as err:
        return HTTPClientError(status_code=400, detail=str(err))

    # forward 412 and 422 from nucliadb to the client
    except PreconditionFailed as err:
        return HTTPClientError(status_code=412, detail=err.message)
    except UnprocessableEntity as err:
        return HTTPClientError(status_code=422, detail=err.message)

    headers = {
        "NUCLIA-LEARNING-ID": ask_result.nuclia_learning_id or "unknown",
        "Access-Control-Expose-Headers": "NUCLIA-LEARNING-ID",
    }
    if x_synchronous:
        return Response(
            content=await ask_result.json(),
            status_code=200,
            headers=headers,
            media_type="application/json",
        )
    else:
        return StreamingResponse(
            content=ask_result.ndjson_stream(),
            status_code=200,
            headers=headers,
            media_type="application/x-ndjson",
        )

    (TextGenerativeResponse,)
    (ReasoningGenerativeResponse,)
    (JSONGenerativeResponse,)
    (MetaGenerativeResponse,)
    (CitationsGenerativeResponse,)
    (FootnoteCitationsGenerativeResponse,)
    (StatusGenerativeResponse,)
    (RerankGenerativeResponse,)
    (ToolsGenerativeResponse,)
    (ConsumptionGenerative,)
    (ImageGenerativeResponse,)


@handled_ask_exceptions
async def create_agentic_response(
    kbid: str,
    account: str,
    internal_nua_api: str,
    agentic_config: AgenticConfigSchema,
    ask_request: AskRequest,
    user_id: str,
    client_type: NucliaDBClientType,
    origin: str,
    x_synchronous: bool,
    global_drivers: dict[str, Driver],
    resource: str | None = None,
    extra_predict_headers: dict[str, str] | None = None,
) -> Response:
    ask_request.max_tokens = parse_max_tokens(ask_request.max_tokens)

    find_results = KnowledgeboxFindResults(resources={})

    queue: asyncio.Queue[AragAnswer] = asyncio.Queue()

    async def predict_answer_stream() -> AsyncGenerator[GenerativeChunk, None]:
        while True:
            answer = await queue.get()
            if answer.operation == AnswerOperation.DONE:
                break

            if answer.operation == AnswerOperation.ERROR:
                raise UnprocessableEntity(
                    message=answer.exception.detail
                    if answer.exception
                    else "Unknown error in agent execution"
                )

            if answer.operation == AnswerOperation.REASONING:
                yield GenerativeChunk(
                    chunk=ReasoningGenerativeResponse(
                        text=answer.reasoning.text if answer.reasoning else ""
                    )
                )

            if answer.operation == AnswerOperation.ANSWER_CHUNK:
                yield GenerativeChunk(
                    chunk=TextGenerativeResponse(
                        text=answer.streaming_response_chunk.text
                        if answer.streaming_response_chunk
                        else ""
                    )
                )

            if answer.operation == AnswerOperation.ANSWER:
                if answer.step:
                    # We need to collect tokens
                    pass

                if answer.possible_answer:
                    # Not usefull for ask endpoint, more for a agent playground where we want to show the final answer separately
                    pass

                if answer.context:
                    # To fill the find payload
                    pass

                if answer.generated_text:
                    # Not usefull for ask endpoint, more for a agent playground where we want to show the final answer separately
                    pass

                yield GenerativeChunk(
                    chunk=TextGenerativeResponse(
                        text=answer.answer if answer.answer else ""
                    )
                )
                yield GenerativeChunk(
                    chunk=CitationsGenerativeResponse(citations=answer.answer_citations)
                )

            if answer.operation == AnswerOperation.AGENT_REQUEST:
                # Not supported by Ask endpoint
                pass

            if answer.operation == AnswerOperation.START:
                # Not supported by Ask endpoint
                pass

    ask_result = AskResult(
        kbid=kbid,
        ask_request=ask_request,
        main_results=find_results,
        nuclia_learning_id=None,
        predict_answer_stream=predict_answer_stream(),
        prequeries_results=None,
        augmented_context=AugmentedContext(),
        search_sdk=rpc.get_sdk("search"),
        debug_chat_model=None,
        best_matches=[],
        metrics=AskMetrics(),
        auditor=auditor,
        prompt_context=PromptContext(),
        prompt_context_order=PromptContextOrder(),
    )

    async def callback(obj: AragAnswer) -> None:
        await queue.put(obj)

    try:
        drivers: Dict[str, Driver]
        retrieval_config: RetrievalAgentConfig
        retrieval_config, drivers = await transform_agentic_config(
            agentic_config,
            global_drivers,
            ask_request,
            resource,
        )

        agent = await RetrievalAgent.from_config_class(retrieval_config)

        nua = AsyncInternalNuaClient(
            kbid=kbid, account=account, url=internal_nua_api
        )  # TODO: pass real URL if needed

        manager = Manager()
        manager.nua = nua  # type: ignore
        manager.drivers = drivers

        state = State(manager=manager, agent=agent)

        session_memory = EphemeralSessionMemory.from_config(
            retrieval_config.memory,
            agent_id=kbid,
            workflow_id="default",
            rules=retrieval_config.rules,
        )
        session_memory.init(uuid4().hex)

        question_memory = session_memory.start_question(
            ask_request.query, streaming=x_synchronous
        )
        question_memory.set_callback_fn(callback)

        question_memory.session.user_info.update(
            {"user_id": user_id, "client_type": client_type, "origin": origin}
        )

        # if headers:
        #     question_memory.headers.update(headers)

        if state.agent is None:
            raise ValueError("Agent could not be initialized")

        await state.agent(
            question_memory,
            state.manager,
        )

    except AnswerJsonSchemaTooLong as err:
        return HTTPClientError(status_code=400, detail=str(err))

    # forward 412 and 422 from nucliadb to the client
    except PreconditionFailed as err:
        return HTTPClientError(status_code=412, detail=err.message)
    except UnprocessableEntity as err:
        return HTTPClientError(status_code=422, detail=err.message)

    headers = {
        "NUCLIA-LEARNING-ID": ask_result.nuclia_learning_id or "unknown",
        "Access-Control-Expose-Headers": "NUCLIA-LEARNING-ID",
    }
    if x_synchronous:
        return Response(
            content=await ask_result.json(),
            status_code=200,
            headers=headers,
            media_type="application/json",
        )
    else:
        return StreamingResponse(
            content=ask_result.ndjson_stream(),
            status_code=200,
            headers=headers,
            media_type="application/x-ndjson",
        )
