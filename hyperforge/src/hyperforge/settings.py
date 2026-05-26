from pydantic import model_validator
from pydantic_settings import BaseSettings

_REDIRECT_PATH = "/api/auth/agent/{agent_id}/workflow/{workflow_id}/session/{session_id}/oauth/{oauth_uuid}/callback"


class OAuthSettings(BaseSettings):
    nuclia_public_url: str = "https://{zone}.nuclia.com"
    nuclia_zone: str = "arag"
    rao_redirect_url: str = ""

    @model_validator(mode="after")
    def _resolve_urls(self) -> "OAuthSettings":
        self.nuclia_public_url = self.nuclia_public_url.format(zone=self.nuclia_zone)
        if not self.rao_redirect_url:
            self.rao_redirect_url = self.nuclia_public_url.rstrip("/") + _REDIRECT_PATH

        return self
