from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    debug: bool = False
    log_level: str = "ERROR"
    http_host: str = "0.0.0.0"
    http_port: int = 8080

    idp_regional_grpc: str = "idp-grpc.idp-regional.svc.cluster.local:9090"
    dummy_idp: bool = False

    sentry_url: Optional[str] = None
    running_environment: str = "stage"
    zone: str = "stashify"
    grpc_port: int = 8030

    valkey_url: str = "redis://arag-valkey-cluster"
    valkey_cluster_mode: bool = False
    answers_subject: str = (
        "arag.{account}.{agent_id}.{workflow_id}.{session}.{question}.answer"
    )
    oauth_subject: str = "arag.{account}.{agent_id}.{workflow_id}.{session}.{question}.oauth.{oauth_uuid}"
    activate_subject: str = "arag.activate"
    pubsub_keepalive_seconds: float = 20

    load_modules: list[str] = [
        "hyperforge_rephrase",
        "hyperforge_nucliadb",
        "hyperforge_summarize",
    ]

    # Hydra settings for MCP Oauth
    hydra_public_url: str = "https://oauth.progress.cloud"
    hydra_scopes_supported: list[str] = ["offline_access", "openid"]

    hyperforge_google_key: Optional[str] = None
    hyperforge_perplexity_key: Optional[str] = None
    internal_nua_api: str = "http://predict.learning.svc.cluster.local:8080"
