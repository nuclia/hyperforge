import json

from google.genai import Client
from google.auth import load_credentials_from_dict  # type: ignore
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
            if driver.config.credentials:
                info = json.loads(driver.config.credentials)
                credentials, project = load_credentials_from_dict(
                    info,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
        else:
            project = driver.config.location
            location = driver.config.location
        client = Client(
            vertexai=driver.config.vertexai,
            credentials=credentials,
            api_key=driver.config.api_key,
            project=project,
            location=location,
        )
        return cls(
            client=client,
            name=driver.name,
            provider=driver.provider,
        )
