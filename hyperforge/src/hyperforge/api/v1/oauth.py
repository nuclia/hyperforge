import logging
from urllib.parse import urlencode

from cryptography.fernet import InvalidToken
from fastapi import Query
from starlette.requests import Request
from starlette.responses import HTMLResponse

from hyperforge.api.settings import Settings
from hyperforge.api.v1.router import router
from hyperforge.api.v1.utils import tracer
from hyperforge_mcp.http import _fingerprint, decrypt_mcp_oauth_state

logger = logging.getLogger(__name__)

RENDER = "<html><body><h1>OAuth Completed</h1><p>You can close this window and return to the application.</p></body></html>"


def _build_oauth_subject(
    settings: Settings,
    account_id: str,
    agent_id: str,
    workflow_id: str,
    session: str,
    question_id: str,
    oauth_uuid: str,
) -> str:
    return settings.oauth_subject.format(
        account=account_id,
        agent_id=agent_id,
        session=session,
        question=question_id,
        oauth_uuid=oauth_uuid,
        workflow_id=workflow_id,
    )


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
    subject = _build_oauth_subject(
        settings,
        account_id,
        agent_id,
        workflow_id,
        session,
        question_id,
        oauth_uuid,
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


@router.get(
    "/api/auth/mcp/callback",
    status_code=200,
    description="Generic MCP OAuth callback (fixed redirect URI, state-routed)",
    tags=["Retrieval Agent"],
    include_in_schema=False,
)
async def mcp_oauth_callback_generic(
    request: Request,
    code: str | None = Query(None, include_in_schema=False),
    state: str | None = Query(None, include_in_schema=False),
    error: str | None = Query(None, include_in_schema=False),
    error_description: str | None = Query(None, include_in_schema=False),
):
    settings: Settings = request.app.settings

    if state is None:
        logger.warning("MCP generic OAuth callback received without state parameter")
        return HTMLResponse(content="Missing OAuth state parameter", status_code=400)

    try:
        routing = decrypt_mcp_oauth_state(state)
    except InvalidToken:
        logger.warning(
            "MCP generic OAuth callback: invalid or expired state (Fernet decryption failed)"
        )
        return HTMLResponse(content="Invalid or expired OAuth state", status_code=400)

    sdk_state = routing.sdk_state or None
    if sdk_state is None:
        logger.warning(
            "MCP generic OAuth callback: decrypted state missing sdk_state field"
        )
        return HTMLResponse(content="Malformed OAuth state", status_code=400)

    subject = _build_oauth_subject(
        settings,
        routing.account_id,
        routing.agent_id,
        routing.workflow_id,
        routing.session_id,
        routing.question_id,
        routing.oauth_uuid,
    )

    payload_data: dict[str, str] = {}
    if code is not None:
        payload_data["code"] = code
    if sdk_state is not None:
        payload_data["state"] = sdk_state
    if error is not None:
        payload_data["error"] = error
    if error_description is not None:
        payload_data["error_description"] = error_description

    payload = urlencode(payload_data)

    with tracer().start_as_current_span("MCP generic OAuth callback"):
        logger.info(
            "MCP generic OAuth callback: agent=%s workflow=%s session=%s question_id=%s oauth_uuid=%s sdk_state_fp=%s has_code=%s has_error=%s",
            routing.agent_id,
            routing.workflow_id,
            routing.session_id,
            routing.question_id,
            routing.oauth_uuid,
            _fingerprint(sdk_state),
            code is not None,
            error is not None,
        )
        await request.app.broker.send_reply(subject, payload)
        logger.info("mcp_oauth send_reply: published to stream %s", subject)

    if error is not None:
        desc = f": {error_description}" if error_description else ""
        return HTMLResponse(
            content=f"OAuth authorization failed ({error}{desc})", status_code=400
        )
    return HTMLResponse(content=RENDER)
