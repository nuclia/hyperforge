import asyncio
import functools
import inspect
import typing
from enum import Enum
from typing import Optional

from starlette.authentication import AuthCredentials, AuthenticationBackend, BaseUser
from starlette.exceptions import HTTPException
from starlette.requests import HTTPConnection, Request
from starlette.responses import RedirectResponse, Response
from starlette.websockets import WebSocket


class User(BaseUser):
    def __init__(self, username: str, security_groups: list[str] | None = None) -> None:
        self.username = username
        self._security_groups = security_groups

    @property
    def is_authenticated(self) -> bool:
        return True

    @property
    def display_name(self) -> str:
        return self.username

    @property
    def security_groups(self) -> list[str] | None:
        return self._security_groups


class RaoAuthenticationBackend(AuthenticationBackend):
    """Authentication backend with a mixture of RAO and NucliaDB auth.

    This mixture is required while migrating /ask endpoint from NucliaDB to RAO,
    as roles are injected by authorizer and it resolves by path instead of path
    and service. Thus, we handle NucliaDB auth headers (X-NUCLIADB-*) as well as
    the regular learning headers (X-STF-*)

    """

    def __init__(self) -> None:
        self.roles_headers = [
            "X-STF-ROLES",
            "X-NUCLIADB-ROLES",
        ]
        self.user_headers = [
            "X-STF-USER",
            "X-NUCLIADB-USER",
        ]
        self.security_groups_headers = ["X-NUCLIADB-SECURITY-GROUPS"]

    async def authenticate(self, conn) -> tuple[AuthCredentials, BaseUser] | None:
        # There are two groups of headers to authenticate: X-STF-* and
        # X-NUCLIADB-*. As authorizer should only resolve to one set of headers,
        # we scan and try to find any of both. While endpoint roles are properly
        # synchronized with authorizer rules, we don't really care which one
        # there is, nor we will mix them

        request = conn
        auth_creds = None
        for roles_header in self.roles_headers:
            if roles_header in request.headers:
                header_roles = request.headers[roles_header]
                roles = header_roles.split(";")
                auth_creds = AuthCredentials(roles)
                break

        if auth_creds is None:
            return None

        user = None
        for user_header in self.user_headers:
            if user_header in request.headers:
                user = request.headers[user_header]

                raw_security_groups: str | None = None
                for security_group_header in self.security_groups_headers:
                    if security_group_header in request.headers:
                        raw_security_groups = request.headers[security_group_header]
                        break

                security_groups: list[str] | None = None
                if raw_security_groups is not None:
                    security_groups = raw_security_groups.split(";")

                user = User(username=user, security_groups=security_groups)
                break

        if user is None:
            user = User(username="Anonymous")

        return auth_creds, user


def has_required_scope(conn: HTTPConnection, scopes: typing.Sequence[str]) -> bool:
    if conn.auth is None or conn.auth.scopes is None:
        raise HTTPException(status_code=403, detail="Missing authorizer headers.")

    for scope in scopes:
        if scope in conn.auth.scopes:
            return True
    return False


def requires(
    scopes: typing.Union[str, typing.Sequence[str]],
    status_code: int = 403,
    redirect: Optional[str] = None,
) -> typing.Callable:
    # As a fastapi requirement, custom Enum classes have to inherit also from
    # string, so we MUST check for Enum before str
    if isinstance(scopes, Enum):
        scopes_list = [scopes.value]
    elif isinstance(scopes, str):
        scopes_list = [scopes]
    elif isinstance(scopes, list):
        scopes_list = [
            scope.value if isinstance(scope, Enum) else scope for scope in scopes
        ]

    def decorator(func: typing.Callable) -> typing.Callable:
        func.__required_scopes__ = scopes_list  # type: ignore
        type = None
        sig = inspect.signature(func)
        for idx, parameter in enumerate(sig.parameters.values()):
            if parameter.name == "request" or parameter.name == "websocket":
                type = parameter.name
                break
        else:
            raise Exception(
                f'No "request" or "websocket" argument on function "{func}"'
            )

        if type == "websocket":
            # Handle websocket functions. (Always async)
            @functools.wraps(func)
            async def websocket_wrapper(
                *args: typing.Any, **kwargs: typing.Any
            ) -> None:
                websocket = kwargs.get("websocket", None)
                assert isinstance(websocket, WebSocket)

                if not has_required_scope(websocket, scopes_list):
                    await websocket.close()
                else:
                    await func(*args, **kwargs)

            return websocket_wrapper

        elif asyncio.iscoroutinefunction(func):
            # Handle async request/response functions.
            @functools.wraps(func)
            async def async_wrapper(
                *args: typing.Any, **kwargs: typing.Any
            ) -> Response:
                request = kwargs.get("request", None)
                assert isinstance(request, Request)

                if not has_required_scope(request, scopes_list):
                    if redirect is not None:
                        return RedirectResponse(
                            url=request.url_for(redirect), status_code=303
                        )
                    raise HTTPException(status_code=status_code)
                return await func(*args, **kwargs)

            return async_wrapper

        else:
            # Handle sync request/response functions.
            @functools.wraps(func)
            def sync_wrapper(*args: typing.Any, **kwargs: typing.Any) -> Response:
                request = kwargs.get("request", args[idx])
                assert isinstance(request, Request)

                if not has_required_scope(request, scopes_list):
                    if redirect is not None:
                        return RedirectResponse(
                            url=request.url_for(redirect), status_code=303
                        )
                    raise HTTPException(status_code=status_code)
                return func(*args, **kwargs)

            return sync_wrapper

    return decorator


def requires_one(
    scopes: typing.Union[str, typing.Sequence[str]],
    status_code: int = 403,
    redirect: Optional[str] = None,
) -> typing.Callable:
    # As a fastapi requirement, custom Enum classes have to inherit also from
    # string, so we MUST check for Enum before str
    if isinstance(scopes, Enum):
        scopes_list = [scopes.value]
    elif isinstance(scopes, str):
        scopes_list = [scopes]
    elif isinstance(scopes, list):
        scopes_list = [
            scope.value if isinstance(scope, Enum) else scope for scope in scopes
        ]

    def decorator(func: typing.Callable) -> typing.Callable:
        func.__required_scopes__ = scopes_list  # type: ignore
        type = None
        sig = inspect.signature(func)
        for idx, parameter in enumerate(sig.parameters.values()):
            if parameter.name == "request" or parameter.name == "websocket":
                type = parameter.name
                break
        else:
            raise Exception(
                f'No "request" or "websocket" argument on function "{func}"'
            )

        if type == "websocket":
            # Handle websocket functions. (Always async)
            @functools.wraps(func)
            async def websocket_wrapper(
                *args: typing.Any, **kwargs: typing.Any
            ) -> None:
                websocket = kwargs.get("websocket", None)
                assert isinstance(websocket, WebSocket)

                if not has_required_scope(websocket, scopes_list):
                    await websocket.close()
                else:
                    await func(*args, **kwargs)

            return websocket_wrapper

        elif asyncio.iscoroutinefunction(func):
            # Handle async request/response functions.
            @functools.wraps(func)
            async def async_wrapper(
                *args: typing.Any, **kwargs: typing.Any
            ) -> Response:
                request = kwargs.get("request", None)
                assert isinstance(request, Request)

                if not has_required_scope(request, scopes_list):
                    if redirect is not None:
                        return RedirectResponse(
                            url=request.url_for(redirect), status_code=303
                        )
                    raise HTTPException(status_code=status_code)
                return await func(*args, **kwargs)

            return async_wrapper

        else:
            # Handle sync request/response functions.
            @functools.wraps(func)
            def sync_wrapper(*args: typing.Any, **kwargs: typing.Any) -> Response:
                request = kwargs.get("request", args[idx])
                assert isinstance(request, Request)

                if not has_required_scope(request, scopes_list):
                    if redirect is not None:
                        return RedirectResponse(
                            url=request.url_for(redirect), status_code=303
                        )
                    raise HTTPException(status_code=status_code)
                return func(*args, **kwargs)

            return sync_wrapper

    return decorator
