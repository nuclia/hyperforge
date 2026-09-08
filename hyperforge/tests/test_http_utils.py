import asyncio
import socket

import pytest

from hyperforge.utils.http import PrivateUrlError, ensure_public_endpoint


@pytest.mark.asyncio
async def test_ensure_public_endpoint_rejects_private_address(monkeypatch):
    async def getaddrinfo(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))]

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", getaddrinfo)

    with pytest.raises(PrivateUrlError):
        await ensure_public_endpoint("http://localhost:8000")


@pytest.mark.asyncio
async def test_ensure_public_endpoint_accepts_public_address(monkeypatch):
    async def getaddrinfo(*args, **kwargs):
        return [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2001:4860:4860::8888", 0))
        ]

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", getaddrinfo)

    await ensure_public_endpoint("grpc.example.com:443")
