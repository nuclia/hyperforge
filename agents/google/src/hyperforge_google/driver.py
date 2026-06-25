from google.genai import Client
from hyperforge.configure import driver
from hyperforge.driver import Driver
from typing_extensions import Self

from hyperforge_google.config import GoogleDriverConfig


@driver(
    id="google",
    title="Google Source",
    description="Source for interacting with the Google API.",
    config_schema=GoogleDriverConfig,
)
class GoogleDriver(Driver):
    client: Client

    @classmethod
    async def init(cls, driver: GoogleDriverConfig) -> Self:
        creds = None
        if driver.config.vertexai:
            from google.auth import load_credentials_from_file

            creds, _ = load_credentials_from_file(driver.config.credentials)
        client = Client(
            vertexai=driver.config.vertexai,
            credentials=creds,
            api_key=driver.config.api_key,
            project=driver.config.project,
            location=driver.config.location,
        )
        return cls(
            client=client,
            name=driver.name,
            provider=driver.provider,
        )
