from typing import TYPE_CHECKING, Annotated, Any, Dict, List
from uuid import UUID

from fastapi import Header
from pydantic import BaseModel, BeforeValidator
from starlette.requests import Request

from hyperforge.api.authentication import requires_one
from hyperforge.api.models import AgentID, DriverID, StashRoles
from hyperforge.api.v1.router import router
from hyperforge.configure import (
    validate_agent_context,
    validate_agent_generation,
    validate_agent_postprocess,
    validate_agent_preprocess,
    validate_driver,
)
from hyperforge.db.agents import AgentManager
from hyperforge.db.encryption import dump_without_encrypted_fields
from hyperforge.driver import DriverConfig
from hyperforge.models import Rules

if TYPE_CHECKING:
    from hyperforge.api.app import HTTPApplication


@router.post(
    "/api/v1/agent/{agent_id}/rules",
    status_code=200,
    description="Set Agent Configuration",
    tags=["Retrieval Agent"],
)
@requires_one([StashRoles.OWNER])
async def set_rules(
    request: Request,
    agent_id: str,
    item: Rules,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    app: HTTPApplication = request.app
    agent_manager: AgentManager = app.agent_manager

    await agent_manager.set_rules(agent_id=agent_id, account=x_stf_account, rules=item)


@router.get(
    "/api/v1/agent/{agent_id}/rules",
    status_code=200,
    description="Set Agent Configuration",
    tags=["Retrieval Agent"],
)
@requires_one([StashRoles.OWNER])
async def get_rules(
    request: Request,
    agent_id: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    app: HTTPApplication = request.app
    agent_manager: AgentManager = app.agent_manager

    return await agent_manager.get_rules(agent_id=agent_id, account=x_stf_account)


@router.post(
    "/api/v1/agent/{agent_id}/drivers",
    status_code=200,
    description="Set Agent Configuration",
    tags=["Retrieval Agent"],
)
@requires_one([StashRoles.OWNER])
async def add_driver(
    request: Request,
    agent_id: str,
    item: Annotated[DriverConfig, BeforeValidator(validate_driver)],
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> DriverID:
    agent_manager: AgentManager = request.app.agent_manager
    driver_id = await agent_manager.add_driver(
        account=x_stf_account,
        agent_id=agent_id,
        config=item,
    )
    return DriverID(id=driver_id)


@router.patch(
    "/api/v1/agent/{agent_id}/driver/{driver}",
    status_code=200,
    description="Set Agent Configuration",
    tags=["Retrieval Agent"],
)
@requires_one([StashRoles.OWNER])
async def patch_driver(
    request: Request,
    agent_id: str,
    driver: str,
    item: Annotated[DriverConfig, BeforeValidator(validate_driver)],
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    agent_manager: AgentManager = request.app.agent_manager

    await agent_manager.patch_driver(
        account=x_stf_account, agent_id=agent_id, config=item, driver=driver
    )


@router.delete(
    "/api/v1/agent/{agent_id}/driver/{driver}",
    status_code=200,
    description="Set Agent Configuration",
    tags=["Retrieval Agent"],
)
@requires_one([StashRoles.OWNER])
async def delete_driver(
    request: Request,
    agent_id: str,
    driver: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    agent_manager: AgentManager = request.app.agent_manager

    return await agent_manager.delete_driver(
        account=x_stf_account, agent_id=agent_id, driver=driver
    )


@router.get(
    "/api/v1/agent/{agent_id}/drivers",
    status_code=200,
    description="Get Agent Configuration",
    tags=["Retrieval Agent"],
)
@requires_one([StashRoles.OWNER])
async def get_drivers(
    request: Request,
    agent_id: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> list[Dict[str, Any]]:
    agent_manager: AgentManager = request.app.agent_manager

    # TODO: This returns DriverConfig while the rest of APIs use DriverConfigs
    # Standardize (into the simpler DriverConfigs?) for consistency
    # Same for the `get_driver` function
    drivers: list[DriverConfig] = await agent_manager.get_drivers(
        account=x_stf_account,
        agent_id=agent_id,
    )
    return [dump_without_encrypted_fields(driver) for driver in drivers]


@router.post(
    "/api/v1/agent/{agent_id}/preprocess",
    status_code=200,
    description="Add PreProcess Agent Configuration",
    tags=["Retrieval Agent"],
)
@requires_one([StashRoles.OWNER])
async def add_preprocess(
    request: Request,
    agent_id: str,
    item: Annotated[BaseModel, BeforeValidator(validate_agent_preprocess)],
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> AgentID:
    agent_manager: AgentManager = request.app.agent_manager

    agent_id = await agent_manager.add_preprocess(
        agent_id=agent_id,
        account=x_stf_account,
        agent=item,
    )
    return AgentID(id=agent_id)


@router.patch(
    "/api/v1/agent/{agent_id}/preprocess/{preprocess}",
    status_code=200,
    description="Set PreProcess Agent Configuration",
    tags=["Retrieval Agent"],
)
@requires_one([StashRoles.OWNER])
async def patch_preprocess(
    request: Request,
    agent_id: str,
    preprocess: str,
    item: Annotated[BaseModel, BeforeValidator(validate_agent_preprocess)],
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    agent_manager: AgentManager = request.app.agent_manager

    await agent_manager.patch_preprocess(
        account=x_stf_account, agent_id=agent_id, preprocess=preprocess, agent=item
    )


@router.delete(
    "/api/v1/agent/{agent_id}/preprocess/{preprocess}",
    status_code=200,
    description="Delete PreProcess Agent Configuration",
    tags=["Retrieval Agent"],
)
@requires_one([StashRoles.OWNER])
async def delete_preprocess(
    request: Request,
    agent_id: str,
    preprocess: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    agent_manager: AgentManager = request.app.agent_manager

    return await agent_manager.delete_preprocess(
        account=x_stf_account, agent_id=agent_id, preprocess=preprocess
    )


@router.get(
    "/api/v1/agent/{agent_id}/preprocess",
    status_code=200,
    description="Set Agent Configuration",
    tags=["Retrieval Agent"],
)
@requires_one([StashRoles.OWNER])
async def get_preprocess(
    request: Request,
    agent_id: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> List[Any]:
    agent_manager: AgentManager = request.app.agent_manager

    return await agent_manager.get_preprocess(
        account=x_stf_account,
        agent_id=agent_id,
    )


@router.post(
    "/api/v1/agent/{agent_id}/generation",
    status_code=200,
    description="Add Generation Agent Configuration",
    tags=["Retrieval Agent"],
)
@requires_one([StashRoles.OWNER])
async def add_generation(
    request: Request,
    agent_id: str,
    item: Annotated[BaseModel, BeforeValidator(validate_agent_generation)],
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> AgentID:
    agent_manager: AgentManager = request.app.agent_manager

    agent_id = await agent_manager.add_generation(
        agent_id=agent_id,
        account=x_stf_account,
        agent=item,
    )
    return AgentID(id=agent_id)


@router.patch(
    "/api/v1/agent/{agent_id}/generation/{generation}",
    status_code=200,
    description="Set PreProcess Agent Configuration",
    tags=["Retrieval Agent"],
)
@requires_one([StashRoles.OWNER])
async def patch_generation(
    request: Request,
    agent_id: str,
    generation: str,
    item: Annotated[BaseModel, BeforeValidator(validate_agent_generation)],
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    agent_manager: AgentManager = request.app.agent_manager
    await agent_manager.patch_generation(
        account=x_stf_account, agent_id=agent_id, generation=generation, agent=item
    )


@router.delete(
    "/api/v1/agent/{agent_id}/generation/{generation}",
    status_code=200,
    description="Delete Generation Agent Configuration",
    tags=["Retrieval Agent"],
)
@requires_one([StashRoles.OWNER])
async def delete_generation(
    request: Request,
    agent_id: str,
    generation: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    agent_manager: AgentManager = request.app.agent_manager

    return await agent_manager.delete_generation(
        account=x_stf_account, agent_id=agent_id, generation=generation
    )


@router.get(
    "/api/v1/agent/{agent_id}/generation",
    status_code=200,
    description="Set Agent Configuration",
    tags=["Retrieval Agent"],
)
@requires_one([StashRoles.OWNER])
async def get_generation(
    request: Request,
    agent_id: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> List[Any]:
    agent_manager: AgentManager = request.app.agent_manager

    return await agent_manager.get_generation(
        account=x_stf_account,
        agent_id=agent_id,
    )


@router.post(
    "/api/v1/agent/{agent_id}/postprocess",
    status_code=200,
    description="Add PostProcess Agent Configuration",
    tags=["Retrieval Agent"],
)
@requires_one([StashRoles.OWNER])
async def add_postprocess(
    request: Request,
    agent_id: str,
    item: Annotated[BaseModel, BeforeValidator(validate_agent_postprocess)],
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> AgentID:
    agent_manager: AgentManager = request.app.agent_manager

    agent_id = await agent_manager.add_postprocess(
        agent_id=agent_id,
        account=x_stf_account,
        agent=item,
    )
    return AgentID(id=agent_id)


@router.patch(
    "/api/v1/agent/{agent_id}/postprocess/{postprocess}",
    status_code=200,
    description="Set PostProcess Agent Configuration",
    tags=["Retrieval Agent"],
)
@requires_one([StashRoles.OWNER])
async def patch_postprocess(
    request: Request,
    agent_id: str,
    postprocess: str,
    item: Annotated[BaseModel, BeforeValidator(validate_agent_postprocess)],
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    agent_manager: AgentManager = request.app.agent_manager

    await agent_manager.patch_postprocess(
        account=x_stf_account, agent_id=agent_id, postprocess=postprocess, agent=item
    )


@router.delete(
    "/api/v1/agent/{agent_id}/postprocess/{postprocess}",
    status_code=200,
    description="Delete PreProcess Agent Configuration",
    tags=["Retrieval Agent"],
)
@requires_one([StashRoles.OWNER])
async def delete_postprocess(
    request: Request,
    agent_id: str,
    postprocess: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    agent_manager: AgentManager = request.app.agent_manager

    return await agent_manager.delete_postprocess(
        account=x_stf_account, agent_id=agent_id, postprocess=postprocess
    )


@router.get(
    "/api/v1/agent/{agent_id}/postprocess",
    status_code=200,
    description="Set Agent Configuration",
    tags=["Retrieval Agent"],
)
@requires_one([StashRoles.OWNER])
async def get_postprocess(
    request: Request,
    agent_id: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> List[Any]:
    agent_manager: AgentManager = request.app.agent_manager

    return await agent_manager.get_postprocess(
        account=x_stf_account,
        agent_id=agent_id,
    )


@router.post(
    "/api/v1/agent/{agent_id}/context",
    status_code=200,
    description="Add Context Agent Configuration",
    tags=["Retrieval Agent"],
    include_in_schema=False,
)
@requires_one([StashRoles.OWNER])
async def add_context(
    request: Request,
    agent_id: str,
    item: Annotated[BaseModel, BeforeValidator(validate_agent_context)],
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> AgentID:
    agent_manager: AgentManager = request.app.agent_manager

    agent_id = await agent_manager.add_context(
        agent_id=agent_id,
        account=x_stf_account,
        agent=item,
    )
    return AgentID(id=agent_id)


@router.patch(
    "/api/v1/agent/{agent_id}/context/{context}",
    status_code=200,
    description="Set Context Agent Configuration",
    tags=["Retrieval Agent"],
    include_in_schema=False,
)
@requires_one([StashRoles.OWNER])
async def patch_context(
    request: Request,
    agent_id: str,
    context: UUID,
    item: Annotated[BaseModel, BeforeValidator(validate_agent_context)],
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    agent_manager: AgentManager = request.app.agent_manager

    await agent_manager.patch_context(
        account=x_stf_account, agent_id=agent_id, context=context, agent=item
    )


@router.delete(
    "/api/v1/agent/{agent_id}/context/{context}",
    status_code=200,
    description="Delete Context Agent Configuration",
    tags=["Retrieval Agent"],
    include_in_schema=False,
)
@requires_one([StashRoles.OWNER])
async def delete_context(
    request: Request,
    agent_id: str,
    context: UUID,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    agent_manager: AgentManager = request.app.agent_manager

    return await agent_manager.delete_context(
        account=x_stf_account, agent_id=agent_id, context=context
    )


@router.get(
    "/api/v1/agent/{agent_id}/context",
    status_code=200,
    description="Get list of Context Agents",
    tags=["Retrieval Agent"],
    include_in_schema=False,
)
@requires_one([StashRoles.OWNER])
async def get_context(
    request: Request,
    agent_id: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> List[Any]:
    agent_manager: AgentManager = request.app.agent_manager

    return await agent_manager.get_context(
        account=x_stf_account,
        agent_id=agent_id,
    )
