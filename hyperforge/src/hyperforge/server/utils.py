from hyperforge.server.cache import Cache
from hyperforge.server.settings import Settings

from hyperforge.memory.memory import (
    BaseSessionMemory,
    EphemeralSessionMemory,
    MemoryConfig,
    NoMemorySessionMemory,
    SessionMemory,
)


async def get_memory(
    settings: Settings,
    session: str,
    cache: Cache,
    config: MemoryConfig,
    agent: str,
    workflow_id: str,
) -> BaseSessionMemory:
    memory: BaseSessionMemory

    if (
        config.nucliadb is not None
        and config.nucliadb.internal
        and settings.internal_nucliadb_url
    ):
        config.nucliadb.url = settings.internal_nucliadb_url
        config.nucliadb.key = None
    elif (
        config.nucliadb is not None
        and config.nucliadb.internal
        and settings.internal_nucliadb_url is None
    ):
        raise Exception("Internal NucliaDB URL not configured")

    if session == "ephemeral":
        memory = NoMemorySessionMemory(
            config=MemoryConfig(nucliadb=None),
            agent_id=agent,
            workflow_id=workflow_id,
            cache=cache,
        )
    elif config.nucliadb is None:
        memory = EphemeralSessionMemory(
            config=config, agent_id=agent, workflow_id=workflow_id, cache=cache
        )
    else:
        memory = SessionMemory(
            config=config,
            agent_id=agent,
            workflow_id=workflow_id,
            cache=cache,
        )

    memory.init(session=session)

    return memory
