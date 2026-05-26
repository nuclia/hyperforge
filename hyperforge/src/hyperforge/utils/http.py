import ipaddress
import socket

import aiodns
from httpx import AsyncClient, AsyncHTTPTransport


class PrivateUrlError(Exception):
    pass


class SafeTransport(AsyncHTTPTransport):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    async def is_private_address(hostname: str) -> bool:
        resolver = aiodns.DNSResolver()
        addr = await resolver.gethostbyname(hostname, socket.AF_INET)
        for address in addr.addresses:
            if ipaddress.ip_address(address).is_private:
                return True
            if (
                not ipaddress.ip_address(address).is_private
                and not ipaddress.ip_address(address).is_global
            ):
                # Matches shared address space (100.64.0.0/10 range) that should be non-routeable and
                # not be used for public internet. Let's consider it internal
                return True
        return False

    async def handle_async_request(self, request):
        url = request.url
        try:
            hostname = url.host if url.host is not None else url.path
            if await self.is_private_address(hostname):
                raise PrivateUrlError("Cannot access private network resources")
        except aiodns.error.DNSError:
            raise PrivateUrlError("Could not lookup hostname")

        return await super().handle_async_request(request)


def safe_http_client(timeout=30, transport_verify: bool = True):
    return AsyncClient(
        timeout=timeout,
        transport=SafeTransport(verify=transport_verify),
    )
