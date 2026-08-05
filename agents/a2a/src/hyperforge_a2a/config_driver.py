from typing import ClassVar, Literal

from hyperforge.driver import DriverConfig, EncryptedPayload
from pydantic import Field, model_validator
from pydantic.config import ConfigDict


class A2AInnerConfig(EncryptedPayload):
    encrypted_fields: ClassVar[list[str]] = [
        "authorization",
        "client_private_key",
    ]

    endpoint: str = Field(..., title="A2A server endpoint")
    use_tls: bool = Field(default=False, title="Use TLS for direct gRPC")
    read_timeout_seconds: int = Field(default=120, gt=0)
    ca_certificate: str | None = Field(default=None, title="TLS CA certificate PEM")
    client_certificate_chain: str | None = Field(
        default=None, title="mTLS client certificate chain PEM"
    )
    client_private_key: str | None = Field(
        default=None, title="mTLS client private key PEM"
    )
    authorization: str | None = Field(default=None, title="Static Authorization header")

    @model_validator(mode="after")
    def validate_tls_settings(self) -> "A2AInnerConfig":
        certificate_configured = self.client_certificate_chain is not None
        private_key_configured = self.client_private_key is not None
        if certificate_configured != private_key_configured:
            raise ValueError(
                "TLS client certificate chain and private key must be configured together"
            )
        if self.endpoint.startswith("http://") and self.use_tls:
            raise ValueError("use_tls requires an HTTPS Agent Card URL")
        if (
            (self.ca_certificate or certificate_configured)
            and not self.use_tls
            and not self.endpoint.startswith("https://")
        ):
            raise ValueError(
                "TLS client credentials require use_tls or an HTTPS Agent Card URL"
            )
        return self


class A2ADriverConfig(DriverConfig[A2AInnerConfig]):
    model_config = ConfigDict(title="A2A")
    provider: Literal["a2a"]
    config: A2AInnerConfig
