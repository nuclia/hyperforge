"""Unit tests for the MCP OAuth generic callback endpoint."""

from unittest.mock import AsyncMock
from urllib.parse import parse_qs

import hyperforge_mcp.http as http_module
import pytest
from cryptography.fernet import Fernet
from hyperforge_mcp.http import (
    MCPOAuthRoutingParams,
    decrypt_mcp_oauth_state,
    encrypt_mcp_oauth_state,
)
from starlette.testclient import TestClient

# ---------------------------------------------------------------------------
# Test Fernet key - injected via env var so EncryptionSettings picks it up.
# ---------------------------------------------------------------------------
_TEST_KEY = Fernet.generate_key().decode()

_ROUTING = MCPOAuthRoutingParams(
    account_id="acct-1",
    agent_id="agent-1",
    workflow_id="wf-1",
    session_id="sess-1",
    question_id="q-1",
    oauth_uuid="q-1",
)

_ROUTING_WITH_SDK_STATE = _ROUTING.model_copy(
    update={"sdk_state": "random-sdk-nonce-abc123"}
)


@pytest.fixture(autouse=True)
def _inject_key(monkeypatch):
    """Inject the test Fernet key and clear the @cache so tests use it."""
    monkeypatch.setenv("ENCRYPTION_SECRET_KEY", _TEST_KEY)
    # Clear @cache so _get_mcp_fernet() returns a Fernet with the test key.

    http_module._get_mcp_fernet.cache_clear()
    yield
    http_module._get_mcp_fernet.cache_clear()


def _make_app():
    """Return a minimal FastAPI app that mounts only the oauth router."""
    from fastapi import FastAPI

    from hyperforge.api.settings import Settings
    from hyperforge.api.v1.router import router

    app = FastAPI()
    app.settings = Settings()
    app.broker = AsyncMock()
    app.include_router(router)
    return app


@pytest.fixture()
def client():
    app = _make_app()
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c, app


# ---------------------------------------------------------------------------
# Unit tests for encrypt_mcp_oauth_state / decrypt_mcp_oauth_state
# ---------------------------------------------------------------------------


def test_encrypt_decrypt_roundtrip():
    """Verify the full model_dump_json -> Fernet -> model_validate_json round-trip works."""
    token = encrypt_mcp_oauth_state(_ROUTING_WITH_SDK_STATE)
    result = decrypt_mcp_oauth_state(token)
    assert result == _ROUTING_WITH_SDK_STATE


# ---------------------------------------------------------------------------
# Tests for the HTTP endpoint
# ---------------------------------------------------------------------------


def _make_state(sdk_state: str = "sdk-nonce-abc") -> str:
    return encrypt_mcp_oauth_state(_ROUTING.model_copy(update={"sdk_state": sdk_state}))


def test_missing_state_returns_400(client):
    tc, _ = client
    resp = tc.get("/api/auth/mcp/callback?code=abc123")
    assert resp.status_code == 400
    assert "Missing OAuth state" in resp.text


def test_invalid_state_returns_400(client):
    tc, _ = client
    resp = tc.get("/api/auth/mcp/callback?state=garbage-token&code=abc123")
    assert resp.status_code == 400
    assert "Invalid or expired" in resp.text


def test_valid_state_publishes_sdk_state_and_returns_200(client):
    tc, app = client
    sdk_nonce = "my-sdk-nonce"
    state = _make_state(sdk_nonce)

    resp = tc.get(f"/api/auth/mcp/callback?state={state}&code=mycode")

    assert resp.status_code == 200
    assert app.broker.send_reply.called
    _, published_payload = app.broker.send_reply.call_args[0]
    params = parse_qs(published_payload)
    assert params["code"] == ["mycode"]
    # The published state must be sdk_state (not the full Fernet token).
    assert params["state"] == [sdk_nonce]


def test_valid_state_can_be_used_twice(client):
    """Fernet tokens are stateless - the same token can theoretically be reused
    within its TTL window.  This tests that the endpoint is idempotent."""
    tc, app = client
    state = _make_state()

    resp1 = tc.get(f"/api/auth/mcp/callback?state={state}&code=mycode")
    resp2 = tc.get(f"/api/auth/mcp/callback?state={state}&code=mycode")
    assert resp1.status_code == 200
    assert resp2.status_code == 200


def test_oauth_error_publishes_and_returns_400(client):
    tc, app = client
    state = _make_state()

    resp = tc.get(
        f"/api/auth/mcp/callback?state={state}&error=access_denied&error_description=User+denied"
    )

    assert resp.status_code == 400
    assert "access_denied" in resp.text
    assert app.broker.send_reply.called
    _, published_payload = app.broker.send_reply.call_args[0]
    params = parse_qs(published_payload)
    assert params["error"] == ["access_denied"]
    assert params["error_description"] == ["User denied"]


def test_oauth_error_without_description(client):
    tc, _ = client
    state = _make_state()

    resp = tc.get(f"/api/auth/mcp/callback?state={state}&error=server_error")
    assert resp.status_code == 400
    assert "server_error" in resp.text


def test_state_encrypted_with_different_key_returns_400(client, monkeypatch):
    """A token encrypted with a different key must be rejected."""

    other_key = Fernet.generate_key().decode()
    monkeypatch.setenv("ENCRYPTION_SECRET_KEY", other_key)
    http_module._get_mcp_fernet.cache_clear()

    # encrypt_mcp_oauth_state now uses the *other* key
    state_with_other_key = _make_state()

    # Reset to original test key for the endpoint
    monkeypatch.setenv("ENCRYPTION_SECRET_KEY", _TEST_KEY)
    http_module._get_mcp_fernet.cache_clear()

    tc, _ = client
    resp = tc.get(f"/api/auth/mcp/callback?state={state_with_other_key}&code=abc")
    assert resp.status_code == 400
    assert "Invalid or expired" in resp.text
