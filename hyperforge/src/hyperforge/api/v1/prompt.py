from fastapi import Header
from hyperforge.db.agents import AgentManager
from starlette.requests import Request

from hyperforge.api.authentication import requires_one
from hyperforge.api.models import PromptID, StashRoles
from hyperforge.api.v1.router import router
from hyperforge.prompts import PromptConfig


@router.post(
    "/api/v1/agent/{agent_id}/prompts",
    status_code=200,
    description="Add Prompt to Agent",
    tags=["Prompt Management"],
)
@requires_one([StashRoles.OWNER])
async def add_prompt(
    request: Request,
    agent_id: str,
    item: PromptConfig,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> PromptID:
    agent_manager: AgentManager = request.app.agent_manager
    prompt_id = await agent_manager.add_prompt(
        account=x_stf_account,
        agent_id=agent_id,
        prompt=item,
    )
    return PromptID(id=prompt_id)


@router.patch(
    "/api/v1/agent/{agent_id}/prompt/{prompt_id}",
    status_code=200,
    description="Update Prompt of Agent",
    tags=["Prompt Management"],
)
@requires_one([StashRoles.OWNER])
async def patch_prompt(
    request: Request,
    agent_id: str,
    prompt_id: str,
    item: PromptConfig,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    agent_manager: AgentManager = request.app.agent_manager

    await agent_manager.set_prompt(
        account=x_stf_account, agent_id=agent_id, prompt=item, prompt_id=prompt_id
    )


@router.delete(
    "/api/v1/agent/{agent_id}/prompt/{prompt_id}",
    status_code=200,
    description="Delete prompt id",
    tags=["Prompt Management"],
)
@requires_one([StashRoles.OWNER])
async def delete_prompt(
    request: Request,
    agent_id: str,
    prompt_id: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    agent_manager: AgentManager = request.app.agent_manager

    return await agent_manager.delete_prompt(
        account=x_stf_account, agent_id=agent_id, prompt_id=prompt_id
    )


@router.get(
    "/api/v1/agent/{agent_id}/prompts",
    status_code=200,
    description="Get Agent Configuration",
    tags=["Prompt Management"],
)
@requires_one([StashRoles.OWNER])
async def get_prompts(
    request: Request,
    agent_id: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> list[PromptConfig]:
    agent_manager: AgentManager = request.app.agent_manager

    # TODO: This returns PromptConfig while the rest of APIs use PromptConfigs
    # Standardize (into the simpler PromptConfigs?) for consistency
    # Same for the `get_prompt` function
    prompts = await agent_manager.get_prompts(
        account=x_stf_account,
        agent_id=agent_id,
    )
    return prompts


@router.get(
    "/api/v1/agent/{agent_id}/prompt/{prompt_id}",
    status_code=200,
    description="Get Agent Configuration",
    tags=["Prompt Management"],
)
@requires_one([StashRoles.OWNER])
async def get_prompt(
    request: Request,
    agent_id: str,
    prompt_id: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> PromptConfig:
    agent_manager: AgentManager = request.app.agent_manager

    # TODO: This returns PromptConfig while the rest of APIs use PromptConfigs
    # Standardize (into the simpler PromptConfigs?) for consistency
    # Same for the `get_prompt` function
    prompt = await agent_manager.get_prompt(
        account=x_stf_account, agent_id=agent_id, prompt_id=prompt_id
    )
    return prompt
