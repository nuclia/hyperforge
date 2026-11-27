from typing import TYPE_CHECKING, Annotated, Any, List
from uuid import UUID

from fastapi import Header, HTTPException
from pydantic import BaseModel, BeforeValidator
from starlette.requests import Request

from hyperforge.api.authentication import requires_one
from hyperforge.api.models import AgentID, SkillID, StashRoles, ToolkitID
from hyperforge.api.v1.router import router
from hyperforge.configure import (
    validate_agent_context,
    validate_agent_generation,
    validate_agent_postprocess,
    validate_agent_preprocess,
    validate_toolkit,
)
from hyperforge.db import exceptions
from hyperforge.db.agents import AgentManager
from hyperforge.harness_sdk import HarnessSkill, ToolkitConfig
from hyperforge.models import Rules
from hyperforge.workflows import (
    WorkflowData,
    WorkflowHarnessUpdate,
    WorkflowInput,
    WorkflowUpdate,
)

if TYPE_CHECKING:
    from hyperforge.api.app import HTTPApplication


async def _not_found_as_404(awaitable):
    try:
        return await awaitable
    except exceptions.NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post(
    "/api/v1/agent/{agent_id}/workflows",
    status_code=200,
    description="Add Workflow Configuration",
    tags=["Workflows"],
)
@requires_one([StashRoles.OWNER])
async def add_workflow(
    request: Request,
    agent_id: str,
    item: WorkflowInput,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    app: HTTPApplication = request.app
    agent_manager: AgentManager = app.agent_manager

    await agent_manager.add_workflow(
        agent_id=agent_id, item=item, account=x_stf_account
    )


@router.get(
    "/api/v1/agent/{agent_id}/workflows",
    status_code=200,
    description="Get Workflow Configuration",
    tags=["Workflows"],
)
@requires_one([StashRoles.OWNER, StashRoles.CONTRIBUTOR, StashRoles.MEMBER])
async def get_workflows(
    request: Request,
    agent_id: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> List[WorkflowData]:
    app: HTTPApplication = request.app
    agent_manager: AgentManager = app.agent_manager

    return await agent_manager.workflows_list(agent_id=agent_id, account=x_stf_account)


@router.patch(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}",
    status_code=200,
    description="Set Workflow Configuration",
    tags=["Workflows"],
)
@requires_one([StashRoles.OWNER])
async def set_workflow(
    request: Request,
    agent_id: str,
    workflow_id: str,
    item: WorkflowUpdate,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    app: HTTPApplication = request.app
    agent_manager: AgentManager = app.agent_manager

    await _not_found_as_404(
        agent_manager.set_workflow(
            workflow_id=workflow_id,
            agent_id=agent_id,
            item=item,
            account=x_stf_account,
        )
    )


@router.delete(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}",
    status_code=200,
    description="Delete Workflow Configuration",
    tags=["Workflows"],
)
@requires_one([StashRoles.OWNER])
async def delete_workflow(
    request: Request,
    agent_id: str,
    workflow_id: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    app: HTTPApplication = request.app
    agent_manager: AgentManager = app.agent_manager

    try:
        await agent_manager.delete_workflow(
            workflow_id=workflow_id,
            agent_id=agent_id,
            account=x_stf_account,
            deleted_by=x_stf_user,
        )
    except exceptions.ProtectedWorkflowError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except exceptions.NotFoundError:
        raise HTTPException(status_code=404, detail="Workflow not found")


@router.post(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/rules",
    status_code=200,
    description="Set Workflow Rules Configuration",
    tags=["Workflows"],
)
@requires_one([StashRoles.OWNER])
async def set_rules(
    request: Request,
    agent_id: str,
    workflow_id: str,
    item: Rules,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    app: HTTPApplication = request.app
    agent_manager: AgentManager = app.agent_manager

    await _not_found_as_404(
        agent_manager.set_workflow_rules(
            agent_id=agent_id,
            workflow_id=workflow_id,
            account=x_stf_account,
            rules=item,
        )
    )


@router.get(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/rules",
    status_code=200,
    description="Get Workflow Rules Configuration",
    tags=["Workflows"],
)
@requires_one([StashRoles.OWNER])
async def get_rules(
    request: Request,
    agent_id: str,
    workflow_id: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    app: HTTPApplication = request.app
    agent_manager: AgentManager = app.agent_manager

    return await _not_found_as_404(
        agent_manager.get_workflow_rules(
            agent_id=agent_id, workflow_id=workflow_id, account=x_stf_account
        )
    )


@router.post(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/preprocess",
    status_code=200,
    description="Add PreProcess Workflows Configuration",
    tags=["Workflow Agents"],
)
@requires_one([StashRoles.OWNER])
async def add_preprocess(
    request: Request,
    agent_id: str,
    workflow_id: str,
    item: Annotated[BaseModel, BeforeValidator(validate_agent_preprocess)],
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> AgentID:
    agent_manager: AgentManager = request.app.agent_manager

    agent_id = await _not_found_as_404(
        agent_manager.add_preprocess(
            agent_id=agent_id,
            account=x_stf_account,
            workflow_id=workflow_id,
            agent=item,
        )
    )
    return AgentID(id=agent_id)


@router.patch(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/preprocess/{preprocess}",
    status_code=200,
    description="Set PreProcess Workflows Configuration",
    tags=["Workflow Agents"],
)
@requires_one([StashRoles.OWNER])
async def patch_preprocess(
    request: Request,
    agent_id: str,
    workflow_id: str,
    preprocess: str,
    item: Annotated[BaseModel, BeforeValidator(validate_agent_preprocess)],
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    agent_manager: AgentManager = request.app.agent_manager

    await _not_found_as_404(
        agent_manager.patch_preprocess(
            account=x_stf_account,
            agent_id=agent_id,
            workflow_id=workflow_id,
            preprocess=preprocess,
            agent=item,
        )
    )


@router.delete(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/preprocess/{preprocess}",
    status_code=200,
    description="Delete PreProcess Workflows Configuration",
    tags=["Workflow Agents"],
)
@requires_one([StashRoles.OWNER])
async def delete_preprocess(
    request: Request,
    agent_id: str,
    preprocess: str,
    workflow_id: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    agent_manager: AgentManager = request.app.agent_manager

    return await _not_found_as_404(
        agent_manager.delete_preprocess(
            account=x_stf_account,
            agent_id=agent_id,
            workflow_id=workflow_id,
            preprocess=preprocess,
        )
    )


@router.get(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/preprocess",
    status_code=200,
    description="Set Workflow Configuration",
    tags=["Workflow Agents"],
)
@requires_one([StashRoles.OWNER])
async def get_preprocess(
    request: Request,
    agent_id: str,
    workflow_id: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> List[Any]:
    agent_manager: AgentManager = request.app.agent_manager

    return await _not_found_as_404(
        agent_manager.get_preprocess(
            account=x_stf_account,
            agent_id=agent_id,
            workflow_id=workflow_id,
        )
    )


@router.post(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/generation",
    status_code=200,
    description="Add Generation Workflows Configuration",
    tags=["Workflow Agents"],
)
@requires_one([StashRoles.OWNER])
async def add_generation(
    request: Request,
    agent_id: str,
    workflow_id: str,
    item: Annotated[BaseModel, BeforeValidator(validate_agent_generation)],
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> AgentID:
    agent_manager: AgentManager = request.app.agent_manager

    agent_id = await _not_found_as_404(
        agent_manager.add_generation(
            agent_id=agent_id,
            account=x_stf_account,
            agent=item,
            workflow_id=workflow_id,
        )
    )
    return AgentID(id=agent_id)


@router.patch(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/generation/{generation}",
    status_code=200,
    description="Set Generation Workflows Configuration",
    tags=["Workflow Agents"],
)
@requires_one([StashRoles.OWNER])
async def patch_generation(
    request: Request,
    agent_id: str,
    workflow_id: str,
    generation: str,
    item: Annotated[BaseModel, BeforeValidator(validate_agent_generation)],
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    agent_manager: AgentManager = request.app.agent_manager

    await _not_found_as_404(
        agent_manager.patch_generation(
            account=x_stf_account,
            agent_id=agent_id,
            generation=generation,
            agent=item,
            workflow_id=workflow_id,
        )
    )


@router.delete(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/generation/{generation}",
    status_code=200,
    description="Delete Generation Workflows Configuration",
    tags=["Workflow Agents"],
)
@requires_one([StashRoles.OWNER])
async def delete_generation(
    request: Request,
    agent_id: str,
    workflow_id: str,
    generation: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    agent_manager: AgentManager = request.app.agent_manager

    return await _not_found_as_404(
        agent_manager.delete_generation(
            account=x_stf_account,
            agent_id=agent_id,
            generation=generation,
            workflow_id=workflow_id,
        )
    )


@router.get(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/generation",
    status_code=200,
    description="Get Generation Workflows Configuration",
    tags=["Workflow Agents"],
)
@requires_one([StashRoles.OWNER])
async def get_generation(
    request: Request,
    agent_id: str,
    workflow_id: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> List[Any]:
    agent_manager: AgentManager = request.app.agent_manager

    return await _not_found_as_404(
        agent_manager.get_generation(
            account=x_stf_account,
            agent_id=agent_id,
            workflow_id=workflow_id,
        )
    )


@router.post(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/postprocess",
    status_code=200,
    description="Add PostProcess Workflows Configuration",
    tags=["Workflow Agents"],
)
@requires_one([StashRoles.OWNER])
async def add_postprocess(
    request: Request,
    agent_id: str,
    workflow_id: str,
    item: Annotated[BaseModel, BeforeValidator(validate_agent_postprocess)],
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> AgentID:
    agent_manager: AgentManager = request.app.agent_manager

    agent_id = await _not_found_as_404(
        agent_manager.add_postprocess(
            agent_id=agent_id,
            account=x_stf_account,
            agent=item,
            workflow_id=workflow_id,
        )
    )
    return AgentID(id=agent_id)


@router.patch(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/postprocess/{postprocess}",
    status_code=200,
    description="Set PostProcess Workflows Configuration",
    tags=["Workflow Agents"],
)
@requires_one([StashRoles.OWNER])
async def patch_postprocess(
    request: Request,
    agent_id: str,
    workflow_id: str,
    postprocess: str,
    item: Annotated[BaseModel, BeforeValidator(validate_agent_postprocess)],
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    agent_manager: AgentManager = request.app.agent_manager

    await _not_found_as_404(
        agent_manager.patch_postprocess(
            account=x_stf_account,
            agent_id=agent_id,
            postprocess=postprocess,
            agent=item,
            workflow_id=workflow_id,
        )
    )


@router.delete(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/postprocess/{postprocess}",
    status_code=200,
    description="Delete PostProcess Workflows Configuration",
    tags=["Workflow Agents"],
)
@requires_one([StashRoles.OWNER])
async def delete_postprocess(
    request: Request,
    agent_id: str,
    workflow_id: str,
    postprocess: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    agent_manager: AgentManager = request.app.agent_manager

    return await _not_found_as_404(
        agent_manager.delete_postprocess(
            account=x_stf_account,
            agent_id=agent_id,
            postprocess=postprocess,
            workflow_id=workflow_id,
        )
    )


@router.get(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/postprocess",
    status_code=200,
    description="Get PostProcess Workflows Configuration",
    tags=["Workflow Agents"],
)
@requires_one([StashRoles.OWNER])
async def get_postprocess(
    request: Request,
    agent_id: str,
    workflow_id: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> List[Any]:
    agent_manager: AgentManager = request.app.agent_manager

    return await _not_found_as_404(
        agent_manager.get_postprocess(
            account=x_stf_account,
            agent_id=agent_id,
            workflow_id=workflow_id,
        )
    )


@router.post(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/context",
    status_code=200,
    description="Add Context Workflows Configuration",
    tags=["Workflow Agents"],
    include_in_schema=False,
)
@requires_one([StashRoles.OWNER])
async def add_context(
    request: Request,
    agent_id: str,
    workflow_id: str,
    item: Annotated[BaseModel, BeforeValidator(validate_agent_context)],
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> AgentID:
    agent_manager: AgentManager = request.app.agent_manager

    agent_id = await _not_found_as_404(
        agent_manager.add_context(
            agent_id=agent_id,
            account=x_stf_account,
            agent=item,
            workflow_id=workflow_id,
        )
    )
    return AgentID(id=agent_id)


@router.patch(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/context/{context}",
    status_code=200,
    description="Set Context Workflows Configuration",
    tags=["Workflow Agents"],
    include_in_schema=False,
)
@requires_one([StashRoles.OWNER])
async def patch_context(
    request: Request,
    agent_id: str,
    workflow_id: str,
    context: UUID,
    item: Annotated[BaseModel, BeforeValidator(validate_agent_context)],
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    agent_manager: AgentManager = request.app.agent_manager

    await _not_found_as_404(
        agent_manager.patch_context(
            account=x_stf_account,
            agent_id=agent_id,
            context=context,
            agent=item,
            workflow_id=workflow_id,
        )
    )


@router.delete(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/context/{context}",
    status_code=200,
    description="Delete Context Workflows Configuration",
    tags=["Workflow Agents"],
    include_in_schema=False,
)
@requires_one([StashRoles.OWNER])
async def delete_context(
    request: Request,
    agent_id: str,
    workflow_id: str,
    context: UUID,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    agent_manager: AgentManager = request.app.agent_manager

    return await _not_found_as_404(
        agent_manager.delete_context(
            account=x_stf_account,
            agent_id=agent_id,
            context=context,
            workflow_id=workflow_id,
        )
    )


@router.get(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/context",
    status_code=200,
    description="Get list of Context Workflows Configuration",
    tags=["Workflow Agents"],
    include_in_schema=False,
)
@requires_one([StashRoles.OWNER])
async def get_context(
    request: Request,
    agent_id: str,
    workflow_id: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> List[Any]:
    agent_manager: AgentManager = request.app.agent_manager

    return await _not_found_as_404(
        agent_manager.get_context(
            account=x_stf_account,
            agent_id=agent_id,
            workflow_id=workflow_id,
        )
    )


@router.get(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/harness",
    status_code=200,
    description="Get Workflow Harness Configuration",
    tags=["Workflows"],
)
@requires_one([StashRoles.OWNER])
async def get_workflow_harness(
    request: Request,
    agent_id: str,
    workflow_id: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> WorkflowHarnessUpdate:
    agent_manager: AgentManager = request.app.agent_manager
    return await _not_found_as_404(
        agent_manager.get_workflow_harness(
            account=x_stf_account,
            agent_id=agent_id,
            workflow_id=workflow_id,
        )
    )


@router.patch(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/harness",
    status_code=200,
    description="Update Workflow Harness Configuration",
    tags=["Workflows"],
)
@requires_one([StashRoles.OWNER])
async def patch_workflow_harness(
    request: Request,
    agent_id: str,
    workflow_id: str,
    item: WorkflowHarnessUpdate,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> None:
    agent_manager: AgentManager = request.app.agent_manager
    await _not_found_as_404(
        agent_manager.set_workflow_harness(
            account=x_stf_account,
            agent_id=agent_id,
            workflow_id=workflow_id,
            item=item,
        )
    )


@router.post(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/toolkits",
    status_code=200,
    description="Attach Toolkit to Workflow",
    tags=["Workflow Toolkits"],
)
@requires_one([StashRoles.OWNER])
async def add_workflow_toolkit(
    request: Request,
    agent_id: str,
    workflow_id: str,
    item: Annotated[ToolkitConfig, BeforeValidator(validate_toolkit)],
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> ToolkitID:
    agent_manager: AgentManager = request.app.agent_manager
    tk_id = await _not_found_as_404(
        agent_manager.add_toolkit(
            account=x_stf_account,
            agent_id=agent_id,
            workflow_id=workflow_id,
            toolkit=item,
        )
    )
    return ToolkitID(id=tk_id)


@router.get(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/toolkits",
    status_code=200,
    description="List Workflow Toolkits",
    tags=["Workflow Toolkits"],
)
@requires_one([StashRoles.OWNER])
async def get_workflow_toolkits(
    request: Request,
    agent_id: str,
    workflow_id: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> List[Any]:
    agent_manager: AgentManager = request.app.agent_manager
    return await _not_found_as_404(
        agent_manager.get_toolkits(
            account=x_stf_account,
            agent_id=agent_id,
            workflow_id=workflow_id,
        )
    )


@router.patch(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/toolkits/{toolkit_id}",
    status_code=200,
    description="Update Workflow Toolkit",
    tags=["Workflow Toolkits"],
)
@requires_one([StashRoles.OWNER])
async def patch_workflow_toolkit(
    request: Request,
    agent_id: str,
    workflow_id: str,
    toolkit_id: UUID,
    item: Annotated[ToolkitConfig, BeforeValidator(validate_toolkit)],
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> None:
    agent_manager: AgentManager = request.app.agent_manager
    await _not_found_as_404(
        agent_manager.patch_toolkit(
            account=x_stf_account,
            agent_id=agent_id,
            workflow_id=workflow_id,
            toolkit_id=toolkit_id,
            toolkit=item,
        )
    )


@router.delete(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/toolkits/{toolkit_id}",
    status_code=200,
    description="Delete Workflow Toolkit",
    tags=["Workflow Toolkits"],
)
@requires_one([StashRoles.OWNER])
async def delete_workflow_toolkit(
    request: Request,
    agent_id: str,
    workflow_id: str,
    toolkit_id: UUID,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> None:
    agent_manager: AgentManager = request.app.agent_manager
    await _not_found_as_404(
        agent_manager.delete_toolkit(
            account=x_stf_account,
            agent_id=agent_id,
            workflow_id=workflow_id,
            toolkit_id=toolkit_id,
        )
    )


@router.post(
    "/api/v1/agent/{agent_id}/skills",
    status_code=200,
    description="Add Skill to Agent",
    tags=["Agent Skills"],
)
@requires_one([StashRoles.OWNER])
async def add_agent_skill(
    request: Request,
    agent_id: str,
    item: HarnessSkill,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> SkillID:
    agent_manager: AgentManager = request.app.agent_manager
    row_id = await agent_manager.add_skill(
        account=x_stf_account, agent_id=agent_id, skill=item
    )
    return SkillID(id=row_id)


@router.get(
    "/api/v1/agent/{agent_id}/skills",
    status_code=200,
    description="List Agent Skills",
    tags=["Agent Skills"],
)
@requires_one([StashRoles.OWNER])
async def get_agent_skills(
    request: Request,
    agent_id: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> List[HarnessSkill]:
    agent_manager: AgentManager = request.app.agent_manager
    return await agent_manager.get_skills(account=x_stf_account, agent_id=agent_id)


@router.patch(
    "/api/v1/agent/{agent_id}/skills/{skill_id}",
    status_code=200,
    description="Update Agent Skill",
    tags=["Agent Skills"],
)
@requires_one([StashRoles.OWNER])
async def patch_agent_skill(
    request: Request,
    agent_id: str,
    skill_id: str,
    item: HarnessSkill,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> None:
    agent_manager: AgentManager = request.app.agent_manager
    await agent_manager.patch_skill(
        account=x_stf_account,
        agent_id=agent_id,
        skill_id=skill_id,
        skill=item,
    )


@router.delete(
    "/api/v1/agent/{agent_id}/skills/{skill_id}",
    status_code=200,
    description="Delete Agent Skill",
    tags=["Agent Skills"],
)
@requires_one([StashRoles.OWNER])
async def delete_agent_skill(
    request: Request,
    agent_id: str,
    skill_id: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> None:
    agent_manager: AgentManager = request.app.agent_manager
    await agent_manager.delete_skill(
        account=x_stf_account, agent_id=agent_id, skill_id=skill_id
    )
