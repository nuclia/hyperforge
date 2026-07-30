from typing import Any

from nuclia.lib.nua import AsyncNuaClient
from nuclia.sdk import AsyncNucliaAuth
from pydantic import BaseModel


class NuaBaseModel(BaseModel):
    async def connect(self):
        raise NotImplementedError("Must implement connect method in subclass")

    @classmethod
    async def connect_internal(cls, kbid: str | None, account: str | None, url: str):
        raise NotImplementedError("Must implement connect_internal method in subclass")


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


class NUAConnection(NuaBaseModel):
    key: str

    async def connect(self):
        na = AsyncNucliaAuth()
        client_id, account_type, account, base_region = await na.validate_nua(self.key)
        if account is None or base_region is None:
            raise Exception("Could not connect to NUA")
        return AsyncNuaClient(token=self.key, account=account, region=base_region)

    @classmethod
    async def connect_internal(cls, kbid: str | None, account: str | None, url: str):
        return AsyncNuaClient.internal(url=url, kbid=kbid, account=account)
