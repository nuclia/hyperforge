from fastapi import Header
from starlette.requests import Request

from hyperforge.api.authentication import requires_one
from hyperforge.api.models import StashRoles
from hyperforge.api.v1.router import router
from hyperforge.configure import (
    get_context_agent_schemas,
    get_driver_agent_schemas,
    get_generation_agent_schemas,
    get_postprocess_agent_schemas,
    get_preprocess_agent_schemas,
)


@router.get(
    "/api/v1/agent/{agent_id}/schema",
    status_code=200,
    description="Get Agent Schema",
    tags=["Retrieval Agent"],
    include_in_schema=False,
)
@requires_one([StashRoles.OWNER])
async def get_schema(
    request: Request,
    agent_id: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    schema = {
        "agents": {
            "context": get_context_agent_schemas(
                running_environment=request.app.settings.running_environment,
                account_id=x_stf_account,
            ),
            "preprocess": get_preprocess_agent_schemas(
                running_environment=request.app.settings.running_environment,
                account_id=x_stf_account,
            ),
            "generation": get_generation_agent_schemas(
                running_environment=request.app.settings.running_environment,
                account_id=x_stf_account,
            ),
            "postprocess": get_postprocess_agent_schemas(
                running_environment=request.app.settings.running_environment,
                account_id=x_stf_account,
            ),
        },
        "drivers": get_driver_agent_schemas(
            running_environment=request.app.settings.running_environment,
            account_id=x_stf_account,
        ),
    }

    return schema
