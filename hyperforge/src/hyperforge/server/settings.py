from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    metrics_port: int = 8090

    debug: bool = False
    log_level: str = "WARNING"

    session_timeout: int = 120
    # Maximum time to answer a question
    question_timeout_seconds: int = 300

    valkey_url: str = "redis://arag-valkey-cluster"
    valkey_cluster_mode: bool = False
    activate_subject: str = "arag.activate"
    answers_subject: str = (
        "arag.{account}.{agent_id}.{workflow_id}.{session}.{question}.answer"
    )
    oauth_subject: str = "arag.{account}.{agent_id}.{workflow_id}.{session}.{question}.oauth.{oauth_uuid}"
    pubsub_keepalive_seconds: float = 20
    pubsub_stream_ttl_seconds: int = 300

    internal_nua_api: str = "http://predict.learning.svc.cluster.local:8080"
    internal_nua: bool = False
    local_openai: Optional[str] = None

    external_nua_api_key: Optional[str] = None

    internal_nucliadb: bool = False
    internal_nucliadb_url: Optional[str] = None

    external_nucliadb_key: Optional[str] = None
    external_nucliadb_url: Optional[str] = None

    sentry_url: Optional[str] = None
    running_environment: str = "stage"
    zone: str = "stashify"
    load_modules: list[str] = []
    auth_success_logo_url: Optional[str] = None

    # Set to True when running as a standalone server (i.e. not inside the
    # full learning cluster).  In standalone mode the agent_id is a human-
    # readable slug from the config file, not a real KB UUID, so we resolve
    # the kbid for internal NUA calls from the account_id request header
    # instead.
    standalone: bool = False
    allow_private_network_endpoints: bool = False

    health_check_enabled: bool = True
