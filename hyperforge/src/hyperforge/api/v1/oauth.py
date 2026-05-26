import logging

from fastapi import Query
from starlette.requests import Request
from starlette.responses import HTMLResponse

from hyperforge.api.settings import Settings
from hyperforge.api.v1.router import router
from hyperforge.api.v1.utils import tracer

logger = logging.getLogger(__name__)

RENDER = "<html><body><h1>OAuth Completed</h1><p>You can close this window and return to the application.</p></body></html>"


@router.get(
    "/api/auth/agent/{agent_id}/workflow/{workflow_id}/session/{session}/oauth/{oauth_uuid}/callback",
    status_code=200,
    description="Get Agent Schema",
    tags=["Retrieval Agent"],
    include_in_schema=False,
)
async def oauth_callback(
    request: Request,
    agent_id: str,
    session: str,
    workflow_id: str,
    oauth_uuid: str,
    question_id: str = Query(..., include_in_schema=False),
    state: str = Query(..., include_in_schema=False),
    account_id: str = Query(..., include_in_schema=False),
):
    """
    Callback from oauth flow on RAO that requires to send creds to websocket
    """
    settings: Settings = request.app.settings
    subject = settings.oauth_subject.format(
        account=account_id,
        agent_id=agent_id,
        session=session,
        question=question_id,
        oauth_uuid=oauth_uuid,
        workflow_id=workflow_id,
    )
    # Request a question
    with tracer().start_as_current_span("Request activation"):
        logger.info(
            "OAuth callback received for agent=%s, session=%s, oauth_uuid=%s, question_id=%s",
            agent_id,
            session,
            oauth_uuid,
            question_id,
        )
        await request.app.broker.send_reply(subject, state)
        logger.info(
            "OAuth callback published to stream %s",
            subject,
        )

    return HTMLResponse(content=RENDER)
