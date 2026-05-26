from pydantic_settings import BaseSettings


class DataManagerSettings(BaseSettings):
    postgresql_dsn: str
    export_read_chunk_size: int = 1024 * 1024  # 1 MB
    export_read_max_size: int = 10 * 1024 * 1024  # 10 MB


class EncryptionSettings(BaseSettings):
    encryption_secret_key: str


class IDPSettings(BaseSettings):
    dummy_idp: bool = False
    idp_regional_grpc: str = "idp-grpc.idp-regional.svc.cluster.local:9090"
