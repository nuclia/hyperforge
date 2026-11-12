import ipaddress
import socket
from collections.abc import AsyncIterator, Iterable
from typing import Any
from urllib.parse import urlsplit

import anyio
import httpcore
from httpcore._backends.base import SOCKET_OPTION, AsyncNetworkStream
from httpx import AsyncClient, AsyncHTTPTransport, Request, Response


class PrivateUrlError(Exception):
    pass


async def ensure_public_endpoint(endpoint: str) -> None:
    parsed = urlsplit(endpoint if "://" in endpoint else f"//{endpoint}")
    if parsed.hostname is None:
        raise PrivateUrlError("A valid endpoint hostname is required")
    await _resolve_public_addresses(parsed.hostname, parsed.port or 443)


class ResponseTooLargeError(Exception):
    pass


def validate_public_http_url(url: str, *, https_only: bool = False) -> str:
    parsed = urlsplit(url)
    allowed_schemes = {"https"} if https_only else {"http", "https"}
    if parsed.scheme.lower() not in allowed_schemes or parsed.hostname is None:
        raise ValueError("URL must use an allowed HTTP scheme and include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Credentials are not allowed in URLs")
    return url


async def _resolve_public_addresses(host: str, port: int) -> list[str]:
    try:
        addresses = await anyio.getaddrinfo(
            host,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
        )
    except OSError as exc:
        raise PrivateUrlError("Could not resolve hostname") from exc

    resolved = list(dict.fromkeys(address[4][0] for address in addresses))
    if not resolved or any(
        not ipaddress.ip_address(address.split("%", 1)[0]).is_global
        for address in resolved
    ):
        raise PrivateUrlError("Cannot access non-public network resources")
    return resolved


class SafeNetworkBackend(httpcore.AsyncNetworkBackend):
    def __init__(self) -> None:
        self._backend = httpcore.AsyncConnectionPool()._network_backend

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[SOCKET_OPTION] | None = None,
    ) -> AsyncNetworkStream:
        addresses = await _resolve_public_addresses(host, port)
        last_error: Exception | None = None
        for address in addresses:
            try:
                return await self._backend.connect_tcp(
                    address,
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[SOCKET_OPTION] | None = None,
    ) -> AsyncNetworkStream:
        raise PrivateUrlError("Unix sockets are not allowed")

    async def sleep(self, seconds: float) -> None:
        await self._backend.sleep(seconds)


class SafeTransport(AsyncHTTPTransport):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pool._network_backend = SafeNetworkBackend()  # type: ignore

    @staticmethod
    async def is_private_address(hostname: str) -> bool:
        try:
            await ensure_public_endpoint(hostname)
        except PrivateUrlError:
            return True
        return False

    async def handle_async_request(self, request: Request) -> Response:
        validate_public_http_url(str(request.url))
        url = request.url
        hostname = url.host if url.host is not None else url.path
        if await self.is_private_address(hostname):
            raise PrivateUrlError("Cannot access private network resources")
        return await super().handle_async_request(request)


async def read_limited_response(response: Response, max_bytes: int) -> bytes:
    content_length = response.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > max_bytes:
                raise ResponseTooLargeError("HTTP response exceeds configured limit")
        except ValueError:
            pass

    content = bytearray()
    async for chunk in response.aiter_bytes():
        if len(content) + len(chunk) > max_bytes:
            raise ResponseTooLargeError("HTTP response exceeds configured limit")
        content.extend(chunk)
    return bytes(content)


async def iter_limited_bytes(
    chunks: AsyncIterator[bytes], max_bytes: int
) -> AsyncIterator[bytes]:
    total = 0
    async for chunk in chunks:
        total += len(chunk)
        if total > max_bytes:
            raise ResponseTooLargeError("HTTP response exceeds configured limit")
        yield chunk


def safe_http_client(timeout: float = 30, transport_verify: bool = True) -> AsyncClient:
    return AsyncClient(
        timeout=timeout,
        transport=SafeTransport(verify=transport_verify),
        trust_env=False,
    )
