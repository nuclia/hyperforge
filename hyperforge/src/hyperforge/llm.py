import json
from collections.abc import AsyncIterator
from typing import Any, Self

import httpx
from nuclia.lib.nua import AsyncNuaClient as NucliaAsyncNuaClient
from nuclia.lib.nua import NuaEndpoint
from nuclia.sdk import AsyncNucliaAuth
from pydantic import BaseModel


class NuaBaseModel(BaseModel):
    async def connect(self):
        raise NotImplementedError("Must implement connect method in subclass")

    @classmethod
    async def connect_internal(cls, kbid: str | None, account: str | None, url: str):
        raise NotImplementedError("Must implement connect_internal method in subclass")


class AsyncNuaClient(NucliaAsyncNuaClient):
    """Hyperforge NUA client, including the chat-completions compatibility API."""

    @classmethod
    def internal(
        cls,
        url: str,
        kbid: str | None = None,
        account: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> Self:
        return cls(
            region=url,
            account=account or "",
            headers=headers,
            endpoint=NuaEndpoint.INTERNAL,
            kbid=kbid,
        )

    async def chat_completions_stream(
        self, payload: dict[str, Any], *, timeout: float = 5 * 60
    ) -> AsyncIterator[dict[str, Any]]:
        path = (
            "/api/internal/predict/compat/chat/completions"
            if self.endpoint == NuaEndpoint.INTERNAL
            else "/api/v1/predict/compat/chat/completions"
        )
        request = {**payload, "stream": True}
        async with self.stream_client.stream(
            "POST",
            f"{self.url}{path}",
            json=request,
            headers={"accept": "text/event-stream"},
            timeout=timeout,
        ) as response:
            if response.status_code != 200:
                detail = (await response.aread()).decode(errors="replace")
                error = httpx.HTTPStatusError(
                    f"Nuclia chat completions API error: {response.status_code} - {detail}",
                    request=response.request,
                    response=response,
                )
                raise error
            async for line in response.aiter_lines():
                line = line.strip()
                if not line or line.startswith(":") or line.startswith("event:"):
                    continue
                if line == "data: [DONE]":
                    return
                if line.startswith("data:"):
                    line = line[5:].lstrip()
                yield json.loads(line)


class NoopNuaClient(AsyncNuaClient):  # pragma: no cover
    """
    A no-op NUA client used when no LLM backend is configured.

    Agents that don't need LLM calls (e.g. the ``static`` context agent) work
    fine with this client.  Any method that actually tries to call the NUA API
    will raise a ``RuntimeError`` with a clear message so users know they need
    to configure an LLM backend.
    """

    def __init__(self) -> None:  # type: ignore[override]
        # Intentionally do NOT call super().__init__() — we have no real
        # token/account/region and don't want side-effects from the parent.
        pass

    def _not_configured(self, method: str) -> None:
        raise RuntimeError(
            f"NoopNuaClient: '{method}' was called but no LLM backend is "
            "configured.  Set EXTERNAL_NUA_API_KEY, INTERNAL_NUA=true, "
            "or LOCAL_OPENAI to enable LLM-based agents."
        )

    async def aclose(self) -> None:
        pass

    async def generate(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        self._not_configured("generate")

    async def chat(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        self._not_configured("chat")

    async def summarize(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        self._not_configured("summarize")

    async def rephrase(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        self._not_configured("rephrase")

    async def sentence(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        self._not_configured("sentence")

    async def tokens(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        self._not_configured("tokens")

    async def query(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        self._not_configured("query")

    async def chat_completions_stream(
        self, payload: dict[str, Any], *, timeout: float = 5 * 60
    ) -> AsyncIterator[dict[str, Any]]:
        self._not_configured("chat_completions_stream")
        if False:
            yield {}


class NUAConnection(NuaBaseModel):
    key: str

    async def connect(self, *, base_url: str | None = None):
        na = AsyncNucliaAuth()
        client_id, account_type, account, base_region = await na.validate_nua(self.key)
        if account is None or base_region is None:
            raise Exception("Could not connect to NUA")
        return AsyncNuaClient(
            token=self.key, account=account, region=base_url or base_region
        )

    @classmethod
    async def connect_internal(cls, kbid: str | None, account: str | None, url: str):
        return AsyncNuaClient.internal(url=url, kbid=kbid, account=account)
