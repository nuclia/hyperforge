import logging
from html import escape
from urllib.parse import urlencode

from cryptography.fernet import InvalidToken
from fastapi import Query
from hyperforge_mcp.http import _fingerprint, decrypt_mcp_oauth_state
from starlette.requests import Request
from starlette.responses import HTMLResponse

from hyperforge.api.settings import Settings
from hyperforge.api.v1.router import router
from hyperforge.api.v1.utils import tracer

logger = logging.getLogger(__name__)

RENDER_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Hyperforge</title>
    <style>
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f5f5f5;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            color: #1a1a1a;
        }
        .card {
            background: #fff;
            border-radius: 16px;
            box-shadow: 0 2px 16px rgba(0,0,0,.08);
            padding: 3rem 3.5rem;
            text-align: center;
            max-width: 420px;
            width: 90%;
        }
        .logo {
            width: 180px;
            margin-bottom: 2rem;
        }
        .brand {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 2rem;
        }
        .icon {
            width: 52px;
            height: 52px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 1.25rem;
            font-size: 1.5rem;
        }
        .icon--success { background: #f0fdf4; color: #16a34a; }
        h1 {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: .5rem;
        }
        p {
            font-size: .925rem;
            color: #6b7280;
            line-height: 1.5;
        }
    </style>
</head>
<body>
    <div class="card">
        __AUTH_SUCCESS_BRANDING__
        <div class="icon icon--success">&#10003;</div>
        <h1>Authentication successful</h1>
        <p>Authentication complete. You can close this window and return to the application.</p>
    </div>
</body>
</html>
"""


def _render_auth_success(settings: Settings) -> str:
    if settings.auth_success_logo_url:
        branding = (
            f'<img class="logo" src="{escape(settings.auth_success_logo_url, quote=True)}" '
            'alt="Hyperforge">'
        )
    else:
        branding = '<div class="brand">Hyperforge</div>'
    return RENDER_TEMPLATE.replace("__AUTH_SUCCESS_BRANDING__", branding)


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

    return HTMLResponse(content=_render_auth_success(settings))


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
    return HTMLResponse(content=_render_auth_success(settings))
