from typing import TYPE_CHECKING

from fastapi import Header, HTTPException, Query
from nucliadb_models import ResourceCreated
from nucliadb_models.resource import Resource, ResourceList
from nucliadb_sdk import NucliaDBAsync
from starlette.requests import Request

from hyperforge import logger
from hyperforge.api.authentication import requires_one

if TYPE_CHECKING:
    from hyperforge.api.app import HTTPApplication
from nucliadb_sdk.v2.exceptions import NotFoundError

from hyperforge.api.models import (
    DEFAULT_RESOURCE_LIST_PAGE_SIZE,
    SessionData,
    StashRoles,
)
from hyperforge.api.session import (
    create_session_resource,
    delete_session_resource,
    get_session_resource,
    list_session_resources,
    update_session_resource,
)
from hyperforge.api.utils import requires_nucliadb_memory
from hyperforge.api.v1.router import router


@router.post(
    "/api/v1/agent/{agent_id}/sessions",
    status_code=200,
    description="Create session",
    tags=["Sessions"],
)
@requires_one([StashRoles.OWNER])
@requires_nucliadb_memory
async def create_session(
    request: Request,
    item: SessionData,
    agent_id: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> ResourceCreated:
    app: HTTPApplication = request.app
    ndb: NucliaDBAsync = app.arag_writer

    try:
        created = await create_session_resource(
            ndb=ndb,
            agent_id=agent_id,
            slug=item.slug,
            title=item.name,
            summary=item.summary,
            data=item.data,
            format=item.format,
        )
        return created
    except Exception as e:
        logger.exception(f"Error creating session on nucliadb: {e}")
        raise HTTPException(status_code=422, detail="Failed to create session")


@router.patch(
    "/api/v1/agent/{agent_id}/session/{session}",
    status_code=200,
    description="Create session",
    tags=["Sessions"],
)
@requires_one([StashRoles.OWNER])
@requires_nucliadb_memory
async def patch_session(
    request: Request,
    agent_id: str,
    session: str,
    item: SessionData,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    app: HTTPApplication = request.app
    ndb: NucliaDBAsync = app.arag_writer
    try:
        await update_session_resource(
            ndb=ndb,
            agent_id=agent_id,
            session_id=session,
            title=item.name,
            summary=item.summary,
            data=item.data,
            format=item.format,
        )
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")


@router.delete(
    "/api/v1/agent/{agent_id}/session/{session}",
    status_code=200,
    description="Delete session",
    tags=["Sessions"],
)
@requires_one([StashRoles.OWNER])
@requires_nucliadb_memory
async def delete_session(
    request: Request,
    agent_id: str,
    session: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    app: HTTPApplication = request.app
    ndb: NucliaDBAsync = app.arag_writer
    try:
        await delete_session_resource(
            ndb=ndb,
            agent_id=agent_id,
            session_id=session,
        )
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")


@router.get(
    "/api/v1/agent/{agent_id}/sessions",
    status_code=200,
    description="Create session",
    tags=["Sessions"],
)
@requires_one([StashRoles.OWNER])
@requires_nucliadb_memory
async def get_sessions(
    request: Request,
    agent_id: str,
    page: int = Query(0, description="Requested page number (0-based)"),
    size: int = Query(DEFAULT_RESOURCE_LIST_PAGE_SIZE, description="Page size"),
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> ResourceList:
    app: HTTPApplication = request.app
    ndb: NucliaDBAsync = app.arag_reader

    return await list_session_resources(
        ndb=ndb,
        agent_id=agent_id,
        page=page,
        size=size,
    )


@router.get(
    "/api/v1/agent/{agent_id}/session/{session}",
    status_code=200,
    description="Create session",
    tags=["Sessions"],
)
@requires_one([StashRoles.OWNER])
@requires_nucliadb_memory
async def get_session(
    request: Request,
    agent_id: str,
    session: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> Resource:
    app: HTTPApplication = request.app
    ndb: NucliaDBAsync = app.arag_reader
    try:
        return await get_session_resource(
            ndb=ndb,
            agent_id=agent_id,
            session_id=session,
            show=["basic", "values"],
        )
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Session not found")
