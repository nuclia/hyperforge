import json
from uuid import UUID

from fastapi import Header, Request, Response
from nucliadb_models.configuration import AskConfig
from nucliadb_models.resource import NucliaDBRoles
from nucliadb_models.search import (
    AskRequest,
    NucliaDBClientType,
    SyncAskResponse,
    parse_max_tokens,
)
from nucliadb_models.security import RequestSecurity
from nucliadb_sdk.v2.exceptions import PreconditionFailed, UnprocessableEntity
from nucliadb_utils.authentication import NucliaUser, requires
from pydantic import ValidationError
from starlette.responses import StreamingResponse

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
