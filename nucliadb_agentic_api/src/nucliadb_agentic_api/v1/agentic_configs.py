from typing import TYPE_CHECKING, cast

from fastapi import Header, HTTPException, Request, Response
from nucliadb_models.resource import NucliaDBRoles
from nucliadb_utils.authentication import requires

from nucliadb_agentic_api import exceptions
from nucliadb_agentic_api.models import AgenticConfigSchema
from nucliadb_agentic_api.v1.router import router

if TYPE_CHECKING:
    from nucliadb_agentic_api.app import HTTPApplication


@router.post(
    "/api/v1/kb/{kbid}/agentic_configs/{agentic_id}",
    status_code=201,
    summary="Create agentic configuration",
    tags=["Agentic configs"],
)
@requires(NucliaDBRoles.MANAGER)
async def create_agentic_config_endpoint(
    request: Request,
    kbid: str,
    agentic_id: str,
    item: AgenticConfigSchema,
    x_nucliadb_account: str = Header(default="", include_in_schema=False),
) -> Response:
    app = cast("HTTPApplication", request.app)
    try:
        await app.agent_manager.create_agentic_config(
            account=x_nucliadb_account,
            kbid=kbid,
            agentic_id=agentic_id,
            config=item,
        )
    except exceptions.Conflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return Response(status_code=201)


@router.get(
    "/api/v1/kb/{kbid}/agentic_configs/{agentic_id}",
    status_code=200,
    summary="Get agentic configuration",
    tags=["Agentic configs"],
    response_model=AgenticConfigSchema,
    response_model_exclude_none=True,
)
@requires(NucliaDBRoles.READER)
async def get_agentic_config_endpoint(
    request: Request,
    kbid: str,
    agentic_id: str,
    x_nucliadb_account: str = Header(default="", include_in_schema=False),
) -> AgenticConfigSchema:
    app = cast("HTTPApplication", request.app)
    try:
        return await app.agent_manager.get_agentic_config(
            account=x_nucliadb_account,
            kbid=kbid,
            agentic_id=agentic_id,
        )
    except exceptions.NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/api/v1/kb/{kbid}/agentic_configs",
    status_code=200,
    summary="List agentic configurations",
    tags=["Agentic configs"],
    response_model=dict[str, AgenticConfigSchema],
    response_model_exclude_none=True,
)
@requires(NucliaDBRoles.READER)
async def list_agentic_configs_endpoint(
    request: Request,
    kbid: str,
    x_nucliadb_account: str = Header(default="", include_in_schema=False),
) -> dict[str, AgenticConfigSchema]:
    app = cast("HTTPApplication", request.app)
    return await app.agent_manager.list_agentic_configs(
        account=x_nucliadb_account,
        kbid=kbid,
    )


@router.patch(
    "/api/v1/kb/{kbid}/agentic_configs/{agentic_id}",
    status_code=204,
    summary="Update agentic configuration",
    tags=["Agentic configs"],
)
@requires(NucliaDBRoles.MANAGER)
async def patch_agentic_config_endpoint(
    request: Request,
    kbid: str,
    agentic_id: str,
    item: AgenticConfigSchema,
    x_nucliadb_account: str = Header(default="", include_in_schema=False),
) -> Response:
    app = cast("HTTPApplication", request.app)
    try:
        await app.agent_manager.patch_agentic_config(
            account=x_nucliadb_account,
            kbid=kbid,
            agentic_id=agentic_id,
            config=item,
        )
    except exceptions.NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(status_code=204)
