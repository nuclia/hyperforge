from hyperforge.configure import driver
from hyperforge.driver import Driver

from hyperforge_a2a.client import build_a2a_client
from hyperforge_a2a.config_driver import A2ADriverConfig, A2AInnerConfig


@driver(
    id="a2a",
    title="A2A Source",
    description="Source for connecting to an Agent2Agent (A2A) server.",
    config_schema=A2ADriverConfig,
)
class A2ADriver(Driver):
    config: A2AInnerConfig

    @classmethod
    async def init(cls, driver: A2ADriverConfig) -> "A2ADriver":
        return cls(
            config=driver.config,
            name=driver.name,
            provider=driver.provider,
        )

    async def client(self, authorization: str | None = None):
        return await build_a2a_client(
            source=self.config.endpoint,
            use_tls=self.config.use_tls,
            ca_certificate=self.config.ca_certificate,
            client_certificate_chain=self.config.client_certificate_chain,
            client_private_key=self.config.client_private_key,
            authorization=authorization or self.config.authorization,
        )
