import asyncio

import httpx
import pytest

from hyperforge.errors import exception_detail


def test_exception_detail_unwraps_nested_exception_groups():
    error = ExceptionGroup(
        "task group",
        [ExceptionGroup("nested", [ValueError("invalid credentials")])],
    )

    assert exception_detail(error) == "invalid credentials"


def test_exception_detail_identifies_authentication_errors():
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(401, request=request)
    error = httpx.HTTPStatusError(
        "Client error '401 Unauthorized'",
        request=request,
        response=response,
    )

    assert exception_detail(ExceptionGroup("task group", [error])) == (
        "Authentication failed (HTTP 401): Client error '401 Unauthorized'"
    )


def test_exception_detail_supports_aiohttp_style_status():
    error = RuntimeError("access denied")
    error.status = 403  # type: ignore[attr-defined]

    assert exception_detail(error) == "Authorization failed (HTTP 403): access denied"


def test_exception_detail_bounds_multiple_errors():
    errors = [RuntimeError(f"error {index}") for index in range(5)]

    assert exception_detail(ExceptionGroup("task group", errors)) == (
        "Multiple errors: error 0; error 1; error 2; and 2 more error(s)"
    )


def test_exception_detail_redacts_secrets():
    error = RuntimeError(
        "request failed: Authorization=Bearer-token password=hunter2 "
        "with Bearer abc.def.ghi"
    )

    assert exception_detail(error) == (
        "request failed: Authorization=[REDACTED] password=[REDACTED] "
        "with Bearer [REDACTED]"
    )


@pytest.mark.parametrize(
    ("detail", "expected"),
    [
        (
            'response={"client_secret": "oauth-secret", "refresh_token":"token"}',
            'response={"client_secret": "[REDACTED]", "refresh_token":"[REDACTED]"}',
        ),
        (
            "request to https://alice:hunter2@example.com failed",
            "request to https://[REDACTED]:[REDACTED]@example.com failed",
        ),
        (
            "Cookie: session=abc; preference=dark",
            "Cookie: [REDACTED]",
        ),
        (
            "Set-Cookie: session=abc; HttpOnly; Secure",
            "Set-Cookie: [REDACTED]",
        ),
        (
            'password="hunter 2"',
            'password="[REDACTED]"',
        ),
        (
            "oauth_token=oauth-value auth_token=auth-value",
            "oauth_token=[REDACTED] auth_token=[REDACTED]",
        ),
        (
            "Authorization: Basic dXNlcjpwYXNzd29yZA==",
            "Authorization: [REDACTED]",
        ),
        (
            'Authorization: "Basic dXNlcjpwYXNzd29yZA=="',
            "Authorization: [REDACTED]",
        ),
    ],
)
def test_exception_detail_redacts_common_structured_secrets(detail, expected):
    assert exception_detail(RuntimeError(detail)) == expected


def test_exception_detail_ignores_cancellation_when_actionable_error_exists():
    error = BaseExceptionGroup(
        "task group",
        [asyncio.CancelledError(), RuntimeError("provider unavailable")],
    )

    assert exception_detail(error) == "provider unavailable"


def test_exception_detail_prioritizes_authentication_errors():
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(401, request=request)
    authentication_error = httpx.HTTPStatusError(
        "invalid token",
        request=request,
        response=response,
    )
    error = ExceptionGroup(
        "task group",
        [RuntimeError("worker stopped"), authentication_error],
    )

    assert exception_detail(error).startswith(
        "Multiple errors: Authentication failed (HTTP 401): invalid token;"
    )
