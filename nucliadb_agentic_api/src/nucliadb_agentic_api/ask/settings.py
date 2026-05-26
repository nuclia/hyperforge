from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    nucliadb_reader_address: str = "http://reader.nucliadb.svc.cluster.local:8080/api"
    nucliadb_search_address: str = "http://search.nucliadb.svc.cluster.local:8080/api"


settings = Settings()
