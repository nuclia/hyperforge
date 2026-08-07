import asyncio
import ipaddress
import socket
from urllib.parse import urlparse

from httpx import AsyncClient, AsyncHTTPTransport


class PrivateUrlError(Exception):
    pass


async def ensure_public_endpoint(endpoint: str) -> None:
    parsed = urlparse(endpoint if "://" in endpoint else f"//{endpoint}")
    if parsed.hostname is None:
        raise PrivateUrlError("A valid endpoint hostname is required")

    try:
        addresses = await asyncio.get_running_loop().getaddrinfo(
            parsed.hostname,
            None,
            type=socket.SOCK_STREAM,
        )
    except OSError as exc:
        raise PrivateUrlError("Could not lookup hostname") from exc

    if any(not ipaddress.ip_address(address[4][0]).is_global for address in addresses):
        raise PrivateUrlError("Cannot access private network resources")


class SafeTransport(AsyncHTTPTransport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    async def is_private_address(hostname: str) -> bool:
        try:
            await ensure_public_endpoint(hostname)
        except PrivateUrlError:
            return True
        return False

    async def handle_async_request(self, request):
        url = request.url
        hostname = url.host if url.host is not None else url.path
        if await self.is_private_address(hostname):
            raise PrivateUrlError("Cannot access private network resources")
        return await super().handle_async_request(request)


def safe_http_client(timeout=30, transport_verify: bool = True):
    return AsyncClient(
        timeout=timeout,
        transport=SafeTransport(verify=transport_verify),
    )
