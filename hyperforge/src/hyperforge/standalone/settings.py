"""
Runtime settings for the standalone arag deployment.

All values can be supplied via:
  - Environment variables (prefixed with ARAG_, e.g. ARAG_EXTERNAL_NUA_API_KEY)
  - A .env file
  - CLI arguments when using the pydantic-settings CLI integration

The agent pipeline definition (drivers, workflows, etc.) lives separately in
a JSON config file pointed to by ``agents_config``.
"""

from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from hyperforge.a2a.settings import A2ASettings


class StandaloneSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ------------------------------------------------------------------
    # Agent config file
    # ------------------------------------------------------------------

    agents_config: Path = Field(
        default=Path("agents_config.yaml"),
        description="Path to the JSON file containing agent definitions.",
    )

    # ------------------------------------------------------------------
    # HTTP server
    # ------------------------------------------------------------------

    host: str = Field(default="0.0.0.0", description="Listen host.")
    port: int = Field(default=8080, description="Listen port.")
    log_level: str = Field(default="INFO", description="Log level (uvicorn + app).")
    debug: bool = Field(default=False, description="Enable debug mode.")
    mcp_force_https_metadata: bool = Field(
        default=True,
        description=(
            "When true, force https URLs in OAuth protected-resource metadata links "
            "emitted by MCP auth challenges and metadata endpoints. "
            "Set false only for local/dev HTTP deployments."
        ),
    )
    auth_success_logo_url: Optional[str] = Field(
        default=None,
        description=(
            "Logo URL shown on the OAuth success page. "
            "When unset, the Hyperforge name is shown instead."
        ),
    )

    # ------------------------------------------------------------------
    # Agent runner
    # ------------------------------------------------------------------

    question_timeout_seconds: int = Field(
        default=300,
        description="Maximum seconds allowed to answer a single question.",
    )
    pubsub_keepalive_seconds: float = Field(
        default=20,
        description="Interval between keepalive pings on the answer stream.",
    )
    pubsub_stream_ttl_seconds: int = Field(
        default=300,
        description="Seconds to retain broker streams in Redis.",
    )
    broker_redis_dsn: Optional[str] = Field(
        default=None,
        description=(
            "Redis DSN for the Pub/Sub broker. If not set, an in-memory broker is used, "
            "which is not suitable for production but fine for local testing and development."
        ),
    )
    broker_redis_activate_subject: str = Field(
        default="arag:activations",
        description="Redis stream subject for agent activations (only used if broker_redis_dsn is set).",
    )
    broker_redis_cluster_mode: bool = Field(
        default=False,
        description="Whether to use Redis Cluster mode (only used if broker_redis_dsn is set).",
    )
    allow_private_network_endpoints: bool = Field(
        default=True,
        description="Allow drivers to connect to private network endpoints.",
    )

    # ------------------------------------------------------------------
    # Optional A2A gRPC server
    # ------------------------------------------------------------------

    a2a_enabled: bool = Field(
        default=False,
        description="Expose the configured standalone agent over A2A gRPC.",
    )
    a2a_grpc_host: str = "0.0.0.0"
    a2a_grpc_port: int = 8034
    a2a_grpc_max_workers: int = 10
    a2a_tls_enabled: bool = False
    a2a_tls_certificate_chain_path: Optional[Path] = None
    a2a_tls_private_key_path: Optional[Path] = None
    a2a_tls_client_ca_path: Optional[Path] = None
    a2a_public_url: Optional[str] = None
    a2a_account: Optional[str] = None
    a2a_agent_id: Optional[str] = None
    a2a_agent_name: str = "Hyperforge"
    a2a_agent_description: str = (
        "Hyperforge agentic RAG exposed over the Agent2Agent (A2A) protocol."
    )
    a2a_agent_version: str = "1.0.0"
    a2a_allowed_forwarded_headers: list[str] = Field(default_factory=list)
    a2a_task_store_prefix: str = "hyperforge:a2a:task"
    a2a_task_ttl_seconds: int = 300

    def a2a_settings(self) -> A2ASettings:
        """Build validated A2A settings using this standalone broker configuration."""
        if not self.a2a_enabled:
            raise ValueError("A2A is not enabled for this standalone application")
        if self.broker_redis_dsn is None:
            raise ValueError("A2A_ENABLED requires BROKER_REDIS_DSN")
        return A2ASettings(
            debug=self.debug,
            log_level=self.log_level,
            a2a_grpc_host=self.a2a_grpc_host,
            a2a_grpc_port=self.a2a_grpc_port,
            a2a_grpc_max_workers=self.a2a_grpc_max_workers,
            a2a_tls_enabled=self.a2a_tls_enabled,
            a2a_tls_certificate_chain_path=self.a2a_tls_certificate_chain_path,
            a2a_tls_private_key_path=self.a2a_tls_private_key_path,
            a2a_tls_client_ca_path=self.a2a_tls_client_ca_path,
            a2a_public_url=self.a2a_public_url,
            a2a_account=self.a2a_account,
            a2a_agent_id=self.a2a_agent_id,
            a2a_agent_name=self.a2a_agent_name,
            a2a_agent_description=self.a2a_agent_description,
            a2a_agent_version=self.a2a_agent_version,
            a2a_allowed_forwarded_headers=self.a2a_allowed_forwarded_headers,
            a2a_task_store_prefix=self.a2a_task_store_prefix,
            a2a_task_ttl_seconds=self.a2a_task_ttl_seconds,
            valkey_url=self.broker_redis_dsn,
            valkey_cluster_mode=self.broker_redis_cluster_mode,
            activate_subject=self.broker_redis_activate_subject,
            pubsub_keepalive_seconds=self.pubsub_keepalive_seconds,
            pubsub_stream_ttl_seconds=self.pubsub_stream_ttl_seconds,
        )

    # ------------------------------------------------------------------
    # NUA / predict engine
    # ------------------------------------------------------------------

    external_nua_api_key: Optional[str] = Field(
        default=None,
        description=(
            "NUA API key from https://nuclia.cloud/user/keys. "
            "Required unless internal_nua=true or local_openai is set."
        ),
        # also allow external_nua_api_key and nua_api_key
        # for backward compatibility with older env var names
        validation_alias=AliasChoices(
            "EXTERNAL_NUA_API_KEY",
            "external_nua_api_key",
            "NUA_API_KEY",
            "nua_api_key",
        ),
    )
    internal_nua: bool = Field(
        default=False,
        description="Connect to an internal NUA service instead of nuclia.cloud.",
    )
    internal_nua_api: str = Field(
        default="http://predict.learning.svc.cluster.local:8080",
        description="Internal NUA service address (only used when internal_nua=true).",
    )
    local_openai: Optional[str] = Field(
        default=None,
        description=(
            "Base URL of a local OpenAI-compatible inference server "
            "(e.g. http://localhost:11434/v1). "
            "When set, requests are routed there instead of nuclia.cloud."
        ),
    )

    in_memory_cache_size: int = 3000
    mcp_max_request_bytes: int = Field(default=1024 * 1024, ge=1)
    mcp_max_response_bytes: int = Field(default=4 * 1024 * 1024, ge=1)

    cors_allow_origin: list[str] = Field(
        default_factory=list,
        description=(
            "List of allowed origins for CORS. Defaults to ['*'] which allows all origins. "
            "In production, it's recommended to set this to the specific origins that should be allowed."
        ),
    )

    ui_admin_username: str = Field(
        default="admin", description="Username for the standalone configuration UI."
    )
    ui_admin_password: SecretStr | None = Field(
        default=None,
        description="Password that enables and protects the standalone configuration UI.",
    )

    enforce_public_urls: bool = Field(
        default=False,
        description="Whether to enforce that only public URLs are allowed.",
    )

    load_modules: list[str] = []
    # ------------------------------------------------------------------
    # Pluggable module classes
    # ------------------------------------------------------------------

    session_cache_class: str | None = Field(
        default=None,
        description="Dotted-name of the session cache class to use.",
    )
    session_cache_size: int = Field(
        default=3000,
        description="Size of the session cache when using a custom session cache class.",
    )
    agent_manager_class: str = Field(
        default="hyperforge.standalone.agent.StaticAgentManager",
        description="Dotted-name of the AgentManager class to use.",
    )
    standalone_application_class: str = Field(
        default="hyperforge.standalone.app.StandaloneApplication",
        description="Dotted-name of the StandaloneApplication class to use.",
    )

    static_folder: str = Field(
        default=str(Path(__file__).parent.parent / "static"),
        description="Path to the static folder for the frontend.",
    )
