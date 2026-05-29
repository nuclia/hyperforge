from pydantic_settings import BaseSettings


class DataManagerSettings(BaseSettings):
    postgresql_dsn: str
    export_read_chunk_size: int = 1024 * 1024  # 1 MB
    export_read_max_size: int = 10 * 1024 * 1024  # 10 MB
