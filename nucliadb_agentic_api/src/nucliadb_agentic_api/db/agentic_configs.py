import datetime

import databases
import sqlalchemy as sa
from hyperforge.database import metadata
from nucliadb_telemetry.utils import get_telemetry, init_telemetry
from sqlalchemy.dialects.postgresql import JSONB

from nucliadb_agentic_api.db.settings import DataManagerSettings
from nucliadb_agentic_api.models import AgenticConfigSchema, AgenticConfiguration
from nucliadb_agentic_api import exceptions

SERVICE_NAME = "AGENTIC_CONFIGS_DB"


def utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


agentic_config_table = sa.Table(
    "agentic_config_table",
    metadata,
    sa.Column("account", sa.String, primary_key=True, nullable=False, index=True),
    sa.Column("kbid", sa.String, primary_key=True, nullable=False, index=True),  # KBID
    sa.Column("agentic_id", sa.String, primary_key=True, nullable=False),  # Agentic ID
    sa.Column("created", sa.DateTime, default=sa.func.now()),
    sa.Column("modified", sa.DateTime, onupdate=sa.func.now()),
    sa.Column("title", sa.String, nullable=True),
    sa.Column("config", JSONB, nullable=False),
)


class AgenticConfigs:
    settings: DataManagerSettings

    def __init__(
        self,
        database: databases.Database,
        settings: DataManagerSettings,
    ):
        self.database = database
        self.settings = settings

    @classmethod
    async def from_settings(
        cls,
        settings: DataManagerSettings,
    ):
        tracer_provider = get_telemetry(SERVICE_NAME)
        if tracer_provider:
            await init_telemetry(tracer_provider)

        database = databases.Database(settings.postgresql_dsn)

        return cls(database=database, settings=settings)

    async def initialize(self):
        await self.database.connect()

    async def finalize(self):
        await self.database.disconnect()

    async def patch_agentic_config(
        self, account: str, kbid: str, agentic_id: str, config: AgenticConfigSchema
    ):
        existing = await self.get_agentic_config(account, kbid, agentic_id)
        if not existing:
            raise exceptions.NotFound("Agentic configuration not found")

        query = (
            sa.update(agentic_config_table)
            .values(
                title=config.title,
                config=config.config,
            )
            .where(
                agentic_config_table.c.account == account,
                agentic_config_table.c.kbid == kbid,
                agentic_config_table.c.agentic_id == agentic_id,
            )
        )
        await self.database.execute(query)

    async def create_agentic_config(
        self, account: str, kbid: str, agentic_id: str, config: AgenticConfigSchema
    ):
        existing = await self.get_agentic_config(account, kbid, agentic_id)
        if existing:
            raise exceptions.Conflict("Agentic configuration already exists")

        query = sa.insert(agentic_config_table).values(
            account=account,
            kbid=kbid,
            agentic_id=agentic_id,
            title=config.title,
            config=config.config,
        )
        await self.database.execute(query)

    async def get_agentic_config(
        self, account: str, kbid: str, agentic_id: str
    ) -> AgenticConfigSchema:
        query = sa.select(agentic_config_table).where(
            agentic_config_table.c.account == account,
            agentic_config_table.c.kbid == kbid,
            agentic_config_table.c.agentic_id == agentic_id,
        )
        row = await self.database.fetch_one(query)
        if not row:
            raise exceptions.NotFound("Agentic configuration not found")

        return AgenticConfigSchema(
            title=row["title"],
            config=AgenticConfiguration.model_validate(row["config"]),
        )
