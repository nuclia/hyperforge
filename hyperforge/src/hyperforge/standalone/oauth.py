import base64
import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.hashes import SHA256, SHA384, SHA512
from starlette.authentication import AuthenticationError

from hyperforge.standalone.config import StandAloneAgentConfig, StandaloneMCPAuthConfig
from hyperforge.standalone.settings import StandaloneSettings
from hyperforge.utils.http import (
    read_limited_response,
    validate_public_http_url,
)

_HASHES = {
    "RS256": SHA256(),
    "RS384": SHA384(),
    "RS512": SHA512(),
}


def force_https_metadata(app: Any) -> bool:
    """Return whether generated MCP OAuth metadata URLs should use HTTPS."""
    for settings_attribute in ("settings", "_standalone_settings"):
        settings = getattr(app, settings_attribute, None)
        if settings is not None and hasattr(settings, "mcp_force_https_metadata"):
            return bool(settings.mcp_force_https_metadata)
    return True


@dataclass
class JWKSCache:
    ttl_seconds: int = 300
    _values: dict[str, tuple[float, dict[str, Any]]] = field(default_factory=dict)
    standalone_settings: StandaloneSettings = field(
        default_factory=lambda: StandaloneSettings()
    )

    async def get(self, url: str) -> dict[str, Any]:
        if self.standalone_settings.enforce_public_urls:
            validate_public_http_url(url, https_only=True)
        now = time.time()
        cached = self._values.get(url)
        if cached is not None:
            expires_at, jwks = cached
            if expires_at > now:
                return jwks

        async with httpx.AsyncClient(timeout=10) as client:
            request = client.build_request("GET", url)
            response = await client.send(request, stream=True, follow_redirects=True)
            try:
                response.raise_for_status()
                content = await read_limited_response(response, 1024 * 1024)
            finally:
                await response.aclose()
        jwks = json.loads(content)
        self._values[url] = (now + self.ttl_seconds, jwks)
        return jwks


def extract_bearer_token(authorization: str | None) -> str:
    if authorization is None:
        raise AuthenticationError("Missing bearer token")
    scheme, _, token = authorization.partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        raise AuthenticationError("Invalid bearer token")
    return token


def get_enabled_mcp_auth(
    agents_cfg: dict[str, Any], agent_id: str
) -> StandaloneMCPAuthConfig | None:
    agent_config = agents_cfg.get(agent_id)
    if not isinstance(agent_config, StandAloneAgentConfig):
        return None

    auth_config = agent_config.mcp_auth
    if auth_config is None or not auth_config.enabled:
        return None
    return auth_config


async def validate_mcp_bearer(
    authorization: str | None,
    auth_config: StandaloneMCPAuthConfig,
    jwks_cache: JWKSCache,
) -> dict[str, Any]:
    token = extract_bearer_token(authorization)
    if auth_config.jwks_url is None:
        raise AuthenticationError("Missing JWKS URL")

    payload = await _validate_jwt(token, auth_config, jwks_cache)
    _validate_scopes(payload, auth_config.required_scopes)
    return payload


async def _validate_jwt(
    token: str,
    auth_config: StandaloneMCPAuthConfig,
    jwks_cache: JWKSCache,
) -> dict[str, Any]:
    try:
        header_raw, payload_raw, signature_raw = token.split(".")
        header = _decode_json(header_raw)
        payload = _decode_json(payload_raw)
        signature = _b64decode(signature_raw)
    except Exception as exc:
        raise AuthenticationError("Malformed bearer token") from exc

    alg = header.get("alg")
    if alg not in _HASHES:
        raise AuthenticationError("Unsupported bearer token algorithm")

    try:
        jwks = await jwks_cache.get(auth_config.jwks_url or "")
    except AuthenticationError:
        raise
    except Exception as exc:
        raise AuthenticationError("Unable to fetch JWKS") from exc
    if not isinstance(jwks, dict):
        raise AuthenticationError("Invalid JWKS")
    key = _select_jwk(jwks, header.get("kid"))
    public_key = _rsa_public_key_from_jwk(key)
    signed_payload = f"{header_raw}.{payload_raw}".encode()

    try:
        public_key.verify(
            signature,
            signed_payload,
            padding.PKCS1v15(),
            _HASHES[alg],
        )
    except Exception as exc:
        raise AuthenticationError("Invalid bearer token signature") from exc

    _validate_claims(payload, auth_config)
    return payload


def _decode_json(value: str) -> dict[str, Any]:
    decoded = _b64decode(value)
    parsed = json.loads(decoded)
    if not isinstance(parsed, dict):
        raise ValueError("JWT part is not an object")
    return parsed


def _b64decode(value: str) -> bytes:
    padded = value + "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(padded.encode())


def _select_jwk(jwks: dict[str, Any], kid: str | None) -> dict[str, Any]:
    keys = jwks.get("keys")
    if not isinstance(keys, list):
        raise AuthenticationError("Invalid JWKS")

    if kid is None:
        if len(keys) != 1:
            raise AuthenticationError("Missing bearer token key id")
        key = keys[0]
        if isinstance(key, dict):
            return key
        raise AuthenticationError("Invalid JWKS")

    for key in keys:
        if isinstance(key, dict) and key.get("kid") == kid:
            return key
    raise AuthenticationError("Bearer token key not found")


def _rsa_public_key_from_jwk(jwk: dict[str, Any]) -> rsa.RSAPublicKey:
    if jwk.get("kty") != "RSA":
        raise AuthenticationError("Unsupported bearer token key type")
    try:
        n = int.from_bytes(_b64decode(jwk["n"]), "big")
        e = int.from_bytes(_b64decode(jwk["e"]), "big")
    except Exception as exc:
        raise AuthenticationError("Invalid bearer token key") from exc
    return rsa.RSAPublicNumbers(e=e, n=n).public_key()


def _validate_claims(
    payload: dict[str, Any], auth_config: StandaloneMCPAuthConfig
) -> None:
    now = time.time()
    exp = payload.get("exp")
    if exp is None:
        raise AuthenticationError("Missing bearer token expiration")
    try:
        exp_value = float(exp)
    except (TypeError, ValueError) as exc:
        raise AuthenticationError("Invalid bearer token expiration") from exc
    if exp_value < now:
        raise AuthenticationError("Expired bearer token")

    nbf = payload.get("nbf")
    if nbf is not None:
        try:
            nbf_value = float(nbf)
        except (TypeError, ValueError) as exc:
            raise AuthenticationError("Invalid bearer token not-before") from exc
        if nbf_value > now:
            raise AuthenticationError("Bearer token is not valid yet")

    if auth_config.issuer is not None and payload.get("iss") != auth_config.issuer:
        raise AuthenticationError("Invalid bearer token issuer")

    if auth_config.audience is not None:
        aud = payload.get("aud")
        audiences = aud if isinstance(aud, list) else [aud]
        if auth_config.audience not in audiences:
            raise AuthenticationError("Invalid bearer token audience")


def _validate_scopes(payload: dict[str, Any], required_scopes: list[str]) -> None:
    if not required_scopes:
        return

    raw_scope = payload.get("scope", "")
    scopes = set(raw_scope.split()) if isinstance(raw_scope, str) else set()
    raw_scp = payload.get("scp", "")
    if isinstance(raw_scp, str):
        scopes.update(raw_scp.split())
    elif isinstance(raw_scp, list):
        scopes.update(str(scope) for scope in raw_scp)
    raw_roles = payload.get("roles", [])
    if isinstance(raw_roles, list):
        scopes.update(str(role) for role in raw_roles)

    if not set(required_scopes).issubset(scopes):
        raise AuthenticationError("Bearer token does not include required scopes")
