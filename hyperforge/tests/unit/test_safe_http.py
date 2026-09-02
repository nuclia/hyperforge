import socket
from unittest.mock import AsyncMock, patch

import pytest

from hyperforge.utils.http import (
    PrivateUrlError,
    ResponseTooLargeError,
    _resolve_public_addresses,
    read_limited_response,
    validate_public_http_url,
)


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "http://user:pass@example.com", "/relative"],
)
def test_validate_public_http_url_rejects_unsafe_urls(url: str):
    with pytest.raises(ValueError):
        validate_public_http_url(url)


def test_validate_public_http_url_requires_https_when_requested():
    with pytest.raises(ValueError):
        validate_public_http_url("http://example.com", https_only=True)
    assert (
        validate_public_http_url("https://example.com", https_only=True)
        == "https://example.com"
    )


@pytest.mark.asyncio
async def test_resolver_rejects_any_non_public_address():
    answers = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
        (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 443, 0, 0)),
    ]
    with (
        patch("anyio.getaddrinfo", AsyncMock(return_value=answers)),
        pytest.raises(PrivateUrlError),
    ):
        await _resolve_public_addresses("example.com", 443)


@pytest.mark.asyncio
async def test_read_limited_response_rejects_large_body():
    import httpx

    response = httpx.Response(200, content=b"12345")
    with pytest.raises(ResponseTooLargeError):
        await read_limited_response(response, 4)
