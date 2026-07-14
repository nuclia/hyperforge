import json

from google.genai import Client
from google.oauth2 import service_account
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
                creds = service_account.Credentials.from_service_account_info(
                    info,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
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
