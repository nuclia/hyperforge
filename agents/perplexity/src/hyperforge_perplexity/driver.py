from hyperforge.configure import driver
from hyperforge.driver import Driver
from perplexity import AsyncPerplexity

from hyperforge_perplexity.config import PerplexityDriverConfig


@driver(
    id="perplexity",
    title="Perplexity Source",
    description="Source for interacting with the Perplexity API.",
    config_schema=PerplexityDriverConfig,
)
class PerplexityDriver(Driver):
    client: AsyncPerplexity
    api_key: str

    @classmethod
    async def init(cls, driver: PerplexityDriverConfig):
        client = AsyncPerplexity(api_key=driver.config.key)

        return cls(
            api_key=driver.config.key,
            client=client,
            name=driver.name,
            provider=driver.provider,
        )
