import base64
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.hashes import SHA256
from httpx import AsyncClient
from httpx._transports.asgi import ASGITransport
from hyperforge.api.v1.mcp_interaction import _prepare_interaction_headers
from hyperforge.standalone import oauth as standalone_oauth
from hyperforge.standalone.app import StandaloneApplication
from hyperforge.standalone.config import StandaloneConfig
from hyperforge.standalone.settings import StandaloneSettings
from starlette.datastructures import Headers

pytestmark = pytest.mark.asyncio


AGENT_ID = "protected-agent"
OPEN_AGENT_ID = "open-agent"
ISSUER = "https://login.example.test/tenant"
AUDIENCE = "api://marklogic"
REQUIRED_SCOPE = "marklogic:read"
PROTECTED_RESOURCE_METADATA_URL = (
    "https://hyperforge.example.test/.well-known/oauth-protected-resource/mcp"
)


@pytest.fixture
def agents_config(load_agents):
    return StandaloneConfig.validate_python(
        {
            AGENT_ID: {
                "title": "Protected Agent",
                "mcp_auth": {
                    "enabled": True,
                    "authorization_server": "https://auth.example.test",
                    "protected_resource_metadata_url": PROTECTED_RESOURCE_METADATA_URL,
                    "protected_resource": AUDIENCE,
                    "scopes_supported": ["openid", "offline_access", REQUIRED_SCOPE],
                    "required_scopes": [REQUIRED_SCOPE],
                    "jwks_url": "https://auth.example.test/jwks",
                    "issuer": ISSUER,
                    "audience": AUDIENCE,
                    "forward_authorization_header": True,
                },
                "workflows": {
                    "default": {
                        "name": "default",
                        "generation": [{"module": "summarize"}],
                    }
                },
            },
            OPEN_AGENT_ID: {
                "title": "Open Agent",
                "workflows": {
                    "default": {
                        "name": "default",
                        "generation": [{"module": "summarize"}],
                    }
                },
            },
        }
    )


@pytest.fixture
def standalone_settings():
    return StandaloneSettings(
        agents_config=Path("/dev/null"),
        external_nua_api_key="dummy",
        debug=False,
    )


@pytest.fixture
async def client(agents_config, standalone_settings):
    app = StandaloneApplication(agents_config, standalone_settings)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        yield client


@pytest.fixture
def signing_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def jwks(signing_key):
    public_numbers = signing_key.public_key().public_numbers()
    return {
        "keys": [
            {
                "kty": "RSA",
                "kid": "test-key",
                "alg": "RS256",
                "n": _b64encode_int(public_numbers.n),
                "e": _b64encode_int(public_numbers.e),
            }
        ]
    }


@pytest.fixture(autouse=True)
def mock_jwks(monkeypatch, jwks):
    async def get(self, url: str):
        return jwks

    monkeypatch.setattr(standalone_oauth.JWKSCache, "get", get)


def make_token(signing_key, kid: str | None = "test-key", **claims_overrides):
    now = int(time.time())
    header = {"alg": "RS256", "typ": "JWT"}
    if kid is not None:
        header["kid"] = kid
    claims = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "exp": now + 300,
        "nbf": now - 10,
        "scp": REQUIRED_SCOPE,
        **claims_overrides,
    }
    header_raw = _b64encode_json(header)
    claims_raw = _b64encode_json(claims)
    signed_payload = f"{header_raw}.{claims_raw}".encode()
    signature = signing_key.sign(signed_payload, padding.PKCS1v15(), SHA256())
    return f"{header_raw}.{claims_raw}.{_b64encode(signature)}"


def _b64encode_json(value: dict) -> str:
    return _b64encode(json.dumps(value, separators=(",", ":")).encode())


def _b64encode_int(value: int) -> str:
    return _b64encode(value.to_bytes((value.bit_length() + 7) // 8, "big"))


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


async def test_extract_bearer_token_strips_extra_token_whitespace():
    token = standalone_oauth.extract_bearer_token("Bearer    token-value   ")

    assert token == "token-value"


async def test_enabled_mcp_auth_requires_authorization_server(load_agents):
    with pytest.raises(ValueError, match="authorization_server"):
        StandaloneConfig.validate_python(
            {
                AGENT_ID: {
                    "mcp_auth": {
                        "enabled": True,
                        "jwks_url": "https://auth.example.test/jwks",
                    },
                    "workflows": {
                        "default": {
                            "name": "default",
                            "generation": [{"module": "summarize"}],
                        }
                    },
                }
            }
        )


async def test_enabled_mcp_auth_requires_jwks_url(load_agents):
    with pytest.raises(ValueError, match="jwks_url"):
        StandaloneConfig.validate_python(
            {
                AGENT_ID: {
                    "mcp_auth": {
                        "enabled": True,
                        "authorization_server": "https://auth.example.test",
                    },
                    "workflows": {
                        "default": {
                            "name": "default",
                            "generation": [{"module": "summarize"}],
                        }
                    },
                }
            }
        )


async def test_mcp_auth_is_optional(client: AsyncClient):
    response = await client.delete(f"/api/v1/agent/{OPEN_AGENT_ID}/session/s1/mcp")

    assert response.status_code == 200


async def test_protected_mcp_requires_bearer(client: AsyncClient):
    response = await client.delete(f"/api/v1/agent/{AGENT_ID}/session/s1/mcp")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == (
        f'Bearer resource_metadata="{PROTECTED_RESOURCE_METADATA_URL}"'
    )


async def test_protected_mcp_accepts_signed_bearer(
    client: AsyncClient, signing_key
):
    response = await client.delete(
        f"/api/v1/agent/{AGENT_ID}/session/s1/mcp",
        headers={"Authorization": f"Bearer {make_token(signing_key)}"},
    )

    assert response.status_code == 200


async def test_protected_mcp_rejects_token_with_wrong_audience(
    client: AsyncClient, signing_key
):
    response = await client.delete(
        f"/api/v1/agent/{AGENT_ID}/session/s1/mcp",
        headers={"Authorization": f"Bearer {make_token(signing_key, aud='api://other')}"},
    )

    assert response.status_code == 401
    assert "audience" in response.text


async def test_protected_mcp_rejects_tampered_token(client: AsyncClient, signing_key):
    token = make_token(signing_key)
    header_raw, claims_raw, signature_raw = token.split(".")
    tampered_claims = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "exp": int(time.time()) + 300,
        "scp": REQUIRED_SCOPE,
        "sub": "tampered",
    }
    tampered_token = (
        f"{header_raw}.{_b64encode_json(tampered_claims)}.{signature_raw}"
    )

    response = await client.delete(
        f"/api/v1/agent/{AGENT_ID}/session/s1/mcp",
        headers={"Authorization": f"Bearer {tampered_token}"},
    )

    assert response.status_code == 401
    assert "signature" in response.text
    assert response.headers["www-authenticate"] == (
        f'Bearer resource_metadata="{PROTECTED_RESOURCE_METADATA_URL}"'
    )


async def test_protected_mcp_rejects_token_without_required_scope(
    client: AsyncClient, signing_key
):
    response = await client.delete(
        f"/api/v1/agent/{AGENT_ID}/session/s1/mcp",
        headers={"Authorization": f"Bearer {make_token(signing_key, scp='other:read')}"},
    )

    assert response.status_code == 401
    assert "scope" in response.text


async def test_protected_mcp_rejects_token_without_exp(
    client: AsyncClient, signing_key
):
    response = await client.delete(
        f"/api/v1/agent/{AGENT_ID}/session/s1/mcp",
        headers={"Authorization": f"Bearer {make_token(signing_key, exp=None)}"},
    )

    assert response.status_code == 401
    assert "expiration" in response.text


async def test_protected_mcp_returns_401_when_jwks_fetch_fails(
    client: AsyncClient, monkeypatch, signing_key
):
    async def fail_get(self, url: str):
        raise RuntimeError("jwks unavailable")

    monkeypatch.setattr(standalone_oauth.JWKSCache, "get", fail_get)

    response = await client.delete(
        f"/api/v1/agent/{AGENT_ID}/session/s1/mcp",
        headers={"Authorization": f"Bearer {make_token(signing_key)}"},
    )

    assert response.status_code == 401
    assert "JWKS" in response.text


async def test_protected_mcp_rejects_token_without_kid_when_jwks_has_multiple_keys(
    client: AsyncClient, monkeypatch, signing_key, jwks
):
    jwks["keys"].append({**jwks["keys"][0], "kid": "rotated-key"})

    async def get(self, url: str):
        return jwks

    monkeypatch.setattr(standalone_oauth.JWKSCache, "get", get)

    response = await client.delete(
        f"/api/v1/agent/{AGENT_ID}/session/s1/mcp",
        headers={"Authorization": f"Bearer {make_token(signing_key, kid=None)}"},
    )

    assert response.status_code == 401
    assert "key id" in response.text


async def test_protected_resource_metadata_uses_agent_oauth_config(
    client: AsyncClient,
):
    response = await client.get(
        f"/.well-known/oauth-protected-resource/api/v1/agent/{AGENT_ID}/session/s1/mcp"
    )

    assert response.status_code == 200
    assert response.json() == {
        "resource": AUDIENCE,
        "scopes_supported": ["openid", "offline_access", REQUIRED_SCOPE],
        "authorization_servers": ["https://auth.example.test"],
    }


async def test_protected_resource_metadata_without_auth_has_no_hydra_fallback(
    client: AsyncClient,
):
    response = await client.get(
        f"/.well-known/oauth-protected-resource/api/v1/agent/{OPEN_AGENT_ID}/session/s1/mcp"
    )

    assert response.status_code == 200
    assert response.json() == {
        "resource": "https://test/api/v1/agent/open-agent/session/s1/mcp",
        "scopes_supported": [],
        "authorization_servers": [],
    }


async def test_prepare_interaction_headers_forwards_authorization(agents_config):
    app = SimpleNamespace(_agents_cfg=agents_config)
    headers = Headers({"Authorization": "Bearer user-token", "X-Other": "value"})

    prepared = _prepare_interaction_headers(app, AGENT_ID, headers)

    assert prepared["authorization"] == "Bearer user-token"
    assert prepared["x-other"] == "value"


async def test_prepare_interaction_headers_can_drop_authorization(agents_config):
    agents_config[AGENT_ID].mcp_auth.forward_authorization_header = False
    app = SimpleNamespace(_agents_cfg=agents_config)
    headers = Headers({"Authorization": "Bearer user-token"})

    prepared = _prepare_interaction_headers(app, AGENT_ID, headers)

    assert "authorization" not in prepared
