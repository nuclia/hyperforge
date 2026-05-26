from typing import TYPE_CHECKING

from starlette.requests import Request

from hyperforge.api.internal.router import router
from hyperforge.api.models import InspectData
from hyperforge.db.agents import AgentManager

if TYPE_CHECKING:
    from hyperforge.api.app import HTTPApplication


@router.get(
    "/api/internal/v1/agent/{kbid}",
    status_code=200,
    description="Report task is done",
    tags=["Task"],
    include_in_schema=False,
)
async def inspect_agent_info(request: Request, kbid: str, account: str) -> InspectData:
    app: HTTPApplication = request.app
    agent_manager: AgentManager = app.agent_manager

    return InspectData(
        contexts=await agent_manager.get_context(account, kbid),
        driver=await agent_manager.get_drivers(account, kbid),
        postprocess=await agent_manager.get_postprocess(account, kbid),
        preprocess=await agent_manager.get_preprocess(account, kbid),
        rules=await agent_manager.get_rules(account, kbid),
    )
