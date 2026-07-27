import datetime
from typing import Any, List
from uuid import UUID

import databases
import sentry_sdk
import sqlalchemy as sa
from cryptography.fernet import Fernet
from fastapi import UploadFile
from nucliadb_telemetry.utils import get_telemetry, init_telemetry
from pydantic import BaseModel, ValidationError
from sqlalchemy.dialects import postgresql as pg_dialect
from sqlalchemy.dialects.postgresql import JSONB

from hyperforge.agent import AgentConfig
from hyperforge.configure import (
    get_agent_config_instance,
    get_driver_config_instance,
    get_driver_config_klass,
)
from hyperforge.database import metadata
from hyperforge.db import exceptions, logger
from hyperforge.db.encryption import (
    decrypt_fields,
    encrypt_fields,
    fernet_key_from_passphrase,
)
from hyperforge.db.settings import DataManagerSettings
from hyperforge.driver import DriverConfig
from hyperforge.models import MemoryConfig, NucliaDBMemoryConfig, Rules
from hyperforge.prompts import PromptArgument, PromptConfig
from hyperforge.retrieval.config import (
    RetrievalAgentConfig,
    RetrievalAgentExportV1,
    retrievalAgentAdapter,
)
from hyperforge.workflows import (
    RetrievalAgent,
    WorkflowData,
    WorkflowInput,
    WorkflowUpdate,
)

SERVICE_NAME = "TASK_MANAGER"
EXPIRATION = 7 * 24
WORKFLOW_PURGE_RETENTION = datetime.timedelta(days=15)


def utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


retrieval_agent_workflow = sa.Table(
    "retrieval_agent_workflow",
    metadata,
    sa.Column("account", sa.String, primary_key=True, nullable=False, index=True),
    sa.Column(
        "agent_id", sa.String, primary_key=True, nullable=False, index=True
    ),  # Agent ID
    sa.Column(
        "workflow_id", sa.String, primary_key=True, nullable=False, index=True
    ),  # Agent ID
    sa.Column("name", sa.String, nullable=False),
    sa.Column("description", sa.String, nullable=True),
    sa.Column("parameters", JSONB, nullable=True),
    sa.Column("required", JSONB, nullable=True),
    sa.Column("rules", JSONB, nullable=True),
    sa.Column("created", sa.DateTime, default=sa.func.now()),
    sa.Column("modified", sa.DateTime, onupdate=sa.func.now()),
    sa.Column("is_deleted", sa.Boolean, nullable=False, default=False, index=True),
    sa.Column("deleted_by", sa.String, nullable=True),
    sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True, index=True),
    sa.ForeignKeyConstraint(
        ["account", "agent_id"],
        ["retrieval_agent_config.account", "retrieval_agent_config.agent_id"],
        onupdate="CASCADE",
        ondelete="CASCADE",
    ),
)

retrieval_agent_prompts = sa.Table(
    "retrieval_agent_prompts",
    metadata,
    sa.Column(
        "id",
        pg_dialect.UUID,
        primary_key=True,
        server_default=sa.func.uuid_generate_v4(),
    ),
    sa.Column(
        "account",
        sa.String,
        nullable=False,
        index=True,
    ),
    sa.Column(
        "agent_id",
        sa.String,
        nullable=False,
        index=True,
    ),
    sa.Column("name", sa.String, nullable=False),
    sa.Column("description", sa.String, nullable=False),
    sa.Column("prompt", sa.String, nullable=False),
    sa.Column("arguments", JSONB, nullable=True),
    sa.Column("icons", JSONB, nullable=True),
    sa.Column("meta", JSONB, nullable=True),
    sa.Column("created", sa.DateTime, default=sa.func.now()),
    sa.Column("modified", sa.DateTime, onupdate=sa.func.now()),
    sa.ForeignKeyConstraint(
        ["account", "agent_id"],
        ["retrieval_agent_config.account", "retrieval_agent_config.agent_id"],
        onupdate="CASCADE",
        ondelete="CASCADE",
    ),
)

retrieval_agent_config = sa.Table(
    "retrieval_agent_config",
    metadata,
    sa.Column("account", sa.String, primary_key=True, nullable=False, index=True),
    sa.Column(
        "agent_id", sa.String, primary_key=True, nullable=False, index=True
    ),  # Agent ID
    sa.Column("rules", JSONB, nullable=False),
    sa.Column("created", sa.DateTime, default=sa.func.now()),
    sa.Column("modified", sa.DateTime, onupdate=sa.func.now()),
    sa.Column("memory", JSONB, nullable=False),
    sa.Column("description", sa.String, nullable=True),
    sa.Column("title", sa.String, nullable=True),
    sa.Column("instructions", sa.String, nullable=True),
)


retrieval_agent_preprocess = sa.Table(
    "retrieval_agent_preprocess",
    metadata,
    sa.Column(
        "id",
        pg_dialect.UUID,
        primary_key=True,
        server_default=sa.func.uuid_generate_v4(),
    ),
    sa.Column(
        "account",
        sa.String,
        nullable=False,
        index=True,
    ),
    sa.Column(
        "agent_id",
        sa.String,
        nullable=False,
        index=True,
    ),
    sa.Column(
        "workflow_id",
        sa.String,
        nullable=False,
        index=True,
    ),
    sa.Column("preprocess", JSONB, nullable=False),
    sa.Column("created", sa.DateTime, default=sa.func.now()),
    sa.Column("modified", sa.DateTime, onupdate=sa.func.now()),
    sa.ForeignKeyConstraint(
        ["account", "agent_id", "workflow_id"],
        [
            "retrieval_agent_workflow.account",
            "retrieval_agent_workflow.agent_id",
            "retrieval_agent_workflow.workflow_id",
        ],
        onupdate="CASCADE",
        ondelete="CASCADE",
    ),
)


retrieval_agent_postprocess = sa.Table(
    "retrieval_agent_postprocess",
    metadata,
    sa.Column(
        "id",
        pg_dialect.UUID,
        primary_key=True,
        server_default=sa.func.uuid_generate_v4(),
    ),
    sa.Column(
        "account",
        sa.String,
        nullable=False,
        index=True,
    ),
    sa.Column(
        "agent_id",
        sa.String,
        nullable=False,
        index=True,
    ),
    sa.Column(
        "workflow_id",
        sa.String,
        nullable=False,
        index=True,
    ),
    sa.Column("postprocess", JSONB, nullable=False),
    sa.Column("created", sa.DateTime, default=sa.func.now()),
    sa.Column("modified", sa.DateTime, onupdate=sa.func.now()),
    sa.ForeignKeyConstraint(
        ["account", "agent_id", "workflow_id"],
        [
            "retrieval_agent_workflow.account",
            "retrieval_agent_workflow.agent_id",
            "retrieval_agent_workflow.workflow_id",
        ],
        onupdate="CASCADE",
        ondelete="CASCADE",
    ),
)

retrieval_agent_context = sa.Table(
    "retrieval_agent_context",
    metadata,
    sa.Column(
        "id",
        pg_dialect.UUID,
        primary_key=True,
        server_default=sa.func.uuid_generate_v4(),
    ),
    sa.Column(
        "account",
        sa.String,
        nullable=False,
        index=True,
    ),
    sa.Column(
        "agent_id",
        sa.String,
        nullable=False,
        index=True,
    ),
    sa.Column(
        "workflow_id",
        sa.String,
        nullable=False,
        index=True,
    ),
    sa.Column("context", JSONB, nullable=False),
    sa.Column("created", sa.DateTime, default=sa.func.now()),
    sa.Column("modified", sa.DateTime, onupdate=sa.func.now()),
    sa.ForeignKeyConstraint(
        ["account", "agent_id", "workflow_id"],
        [
            "retrieval_agent_workflow.account",
            "retrieval_agent_workflow.agent_id",
            "retrieval_agent_workflow.workflow_id",
        ],
        onupdate="CASCADE",
        ondelete="CASCADE",
    ),
)


retrieval_agent_generation = sa.Table(
    "retrieval_agent_generation",
    metadata,
    sa.Column(
        "id",
        pg_dialect.UUID,
        primary_key=True,
        server_default=sa.func.uuid_generate_v4(),
    ),
    sa.Column(
        "account",
        sa.String,
        nullable=False,
        index=True,
    ),
    sa.Column(
        "agent_id",
        sa.String,
        nullable=False,
        index=True,
    ),
    sa.Column(
        "workflow_id",
        sa.String,
        nullable=False,
        index=True,
    ),
    sa.Column("generation", JSONB, nullable=False),
    sa.Column("created", sa.DateTime, default=sa.func.now()),
    sa.Column("modified", sa.DateTime, onupdate=sa.func.now()),
    sa.ForeignKeyConstraint(
        ["account", "agent_id", "workflow_id"],
        [
            "retrieval_agent_workflow.account",
            "retrieval_agent_workflow.agent_id",
            "retrieval_agent_workflow.workflow_id",
        ],
        onupdate="CASCADE",
        ondelete="CASCADE",
    ),
)

retrieval_agents_drivers = sa.Table(
    "retrieval_agents_drivers",
    metadata,
    sa.Column(
        "id",
        pg_dialect.UUID,
        primary_key=True,
        server_default=sa.func.uuid_generate_v4(),
    ),
    sa.Column("account", sa.String, nullable=False),
    sa.Column("agent_id", sa.String, nullable=False),
    sa.Column("driver", sa.String, nullable=False),
    sa.Column("provider", sa.String, nullable=False),
    sa.Column("identifier", sa.String, nullable=False),
    sa.Column("config", JSONB, nullable=False),
    sa.Column("created", sa.DateTime, default=sa.func.now()),
    sa.Column("modified", sa.DateTime, onupdate=sa.func.now()),
    sa.ForeignKeyConstraint(
        ["account", "agent_id"],
        ["retrieval_agent_config.account", "retrieval_agent_config.agent_id"],
        onupdate="CASCADE",
        ondelete="CASCADE",
    ),
    sa.UniqueConstraint("account", "agent_id", "identifier"),
)


class AgentManager:
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

    async def patch_driver(
        self,
        account: str,
        agent_id: str,
        driver: str,
        config: DriverConfig,
    ):
        try:
            previous_config = await self.get_driver(account, agent_id, driver)
        except exceptions.DriverNotFoundError:
            # No previous config for this driver found, nothing to update
            return
        updated_config: DriverConfig = update_driver_config(config, previous_config)
        statement = (
            retrieval_agents_drivers.update()
            .where(
                retrieval_agents_drivers.c.account == account,
            )
            .where(retrieval_agents_drivers.c.agent_id == agent_id)
            .where(retrieval_agents_drivers.c.id == driver)
            .values(
                driver=updated_config.name,
                config=encrypt_fields(updated_config.config),
            )
        )

        await self.database.execute(statement)

    async def add_driver(
        self,
        agent_id: str,
        account: str,
        config: DriverConfig,
    ):
        statement = retrieval_agents_drivers.insert().values(
            account=account,
            agent_id=agent_id,
            driver=config.name,
            provider=config.provider,
            identifier=config.identifier,
            config=encrypt_fields(config.config),
        )

        result = await self.database.execute(statement)

        return str(result)

    async def delete_driver(self, account: str, agent_id: str, driver: str):
        statement = (
            retrieval_agents_drivers.delete()
            .where(retrieval_agents_drivers.c.account == account)
            .where(retrieval_agents_drivers.c.agent_id == agent_id)
            .where(retrieval_agents_drivers.c.id == driver)
        )
        await self.database.execute(statement)

    async def get_driver(
        self, account: str, agent_id: str, driver: str
    ) -> DriverConfig:
        statement = (
            retrieval_agents_drivers.select()
            .where(retrieval_agents_drivers.c.account == account)
            .where(retrieval_agents_drivers.c.agent_id == agent_id)
            .where(retrieval_agents_drivers.c.id == driver)
        )
        result = await self.database.fetch_one(statement)
        if result is None:
            raise exceptions.DriverNotFoundError()

        try:
            config_class = get_driver_config_klass(result["provider"])
        except Exception as e:
            logger.warning(
                f"Driver provider '{result['provider']}' is not registered, treating as not found"
            )
            sentry_sdk.capture_exception(e)
            raise exceptions.DriverNotFoundError()
        driver_config = config_class.model_validate(
            {
                "id": str(result["id"]),
                "name": result["driver"],
                "identifier": result["identifier"],
                "provider": result["provider"],
                "config": result["config"],
            }
        )
        decrypt_fields(driver_config.config)
        return driver_config

    async def get_drivers(self, account: str, agent_id: str) -> List[DriverConfig]:
        statement = (
            retrieval_agents_drivers.select()
            .where(retrieval_agents_drivers.c.account == account)
            .where(retrieval_agents_drivers.c.agent_id == agent_id)
        )
        results = await self.database.fetch_all(statement)
        drivers = []
        for result in results:
            try:
                config_class = get_driver_config_klass(result["provider"])
            except Exception as e:
                logger.warning(
                    f"Skipping driver with unregistered provider '{result['provider']}' "
                    f"for agent {agent_id}"
                )
                sentry_sdk.capture_exception(e)
                continue
            driver = config_class.model_validate(
                {
                    "id": str(result["id"]),
                    "name": result["driver"],
                    "identifier": result["identifier"],
                    "provider": result["provider"],
                    "config": result["config"],
                }
            )
            decrypt_fields(driver.config)
            drivers.append(driver)
        return drivers

    async def add_agent(
        self, account: str, agent_id: str, memory: MemoryConfig, rules: Rules
    ):
        statement = retrieval_agent_config.insert().values(
            account=account,
            agent_id=agent_id,
            rules=rules.model_dump(),
            memory=memory.model_dump(),
        )

        await self.database.execute(statement)

        statement = retrieval_agent_workflow.insert().values(
            account=account,
            agent_id=agent_id,
            workflow_id="default",
            name="default",
            description="Default workflow",
            parameters={},
            rules=Rules(rules=[]).model_dump(),
            required=[],
            is_deleted=False,
        )

        await self.database.execute(statement)

    async def delete_agent(self, account: str, agent_id: str):
        statement = (
            retrieval_agent_config.delete()
            .where(retrieval_agent_config.c.agent_id == agent_id)
            .where(retrieval_agent_config.c.account == account)
        )

        await self.database.execute(statement)

    async def add_workflow(self, account: str, agent_id: str, item: WorkflowInput):
        statement = retrieval_agent_workflow.insert().values(
            account=account,
            agent_id=agent_id,
            workflow_id=item.id,
            name=item.name,
            description=item.description,
            parameters=item.parameters,
            required=item.required,
            rules=item.rules.model_dump()
            if item.rules
            else Rules(rules=[]).model_dump(),
            is_deleted=False,
        )
        await self.database.execute(statement)

    def _active_workflow_condition(self):
        return retrieval_agent_workflow.c.is_deleted.is_(False)

    async def ensure_workflow_active(
        self, account: str, agent_id: str, workflow_id: str
    ):
        statement = (
            retrieval_agent_workflow.select()
            .where(retrieval_agent_workflow.c.account == account)
            .where(retrieval_agent_workflow.c.agent_id == agent_id)
            .where(retrieval_agent_workflow.c.workflow_id == workflow_id)
            .where(self._active_workflow_condition())
        )
        result = await self.database.fetch_one(statement)
        if result is None:
            raise exceptions.NotFoundError("Workflow not found")

    async def set_workflow(
        self, account: str, agent_id: str, workflow_id: str, item: WorkflowUpdate
    ):
        await self.ensure_workflow_active(account, agent_id, workflow_id)
        statement = (
            retrieval_agent_workflow.update()
            .where(retrieval_agent_workflow.c.account == account)
            .where(retrieval_agent_workflow.c.agent_id == agent_id)
            .where(retrieval_agent_workflow.c.workflow_id == workflow_id)
            .where(self._active_workflow_condition())
            .values(
                name=item.name,
                description=item.description,
                parameters=item.parameters,
                required=item.required,
                rules=item.rules.model_dump()
                if item.rules
                else Rules(rules=[]).model_dump(),
            )
        )
        await self.database.execute(statement)

    async def delete_workflow(
        self, account: str, agent_id: str, workflow_id: str, deleted_by: str
    ):
        if workflow_id == "default":
            raise exceptions.ProtectedWorkflowError(
                "Default workflow cannot be deleted"
            )

        await self.ensure_workflow_active(account, agent_id, workflow_id)

        statement = (
            retrieval_agent_workflow.update()
            .where(retrieval_agent_workflow.c.account == account)
            .where(retrieval_agent_workflow.c.agent_id == agent_id)
            .where(retrieval_agent_workflow.c.workflow_id == workflow_id)
            .where(self._active_workflow_condition())
            .values(is_deleted=True, deleted_by=deleted_by, deleted_at=utc_now())
        )
        await self.database.execute(statement)

    async def get_expired_deleted_workflows(
        self, older_than: datetime.timedelta = WORKFLOW_PURGE_RETENTION
    ):
        threshold = utc_now() - older_than
        statement = (
            retrieval_agent_workflow.select()
            .where(retrieval_agent_workflow.c.is_deleted.is_(True))
            .where(retrieval_agent_workflow.c.deleted_at < threshold)
        )
        return await self.database.fetch_all(statement)

    async def purge_deleted_workflow(
        self, account: str, agent_id: str, workflow_id: str
    ):
        statement = (
            retrieval_agent_workflow.delete()
            .where(retrieval_agent_workflow.c.account == account)
            .where(retrieval_agent_workflow.c.agent_id == agent_id)
            .where(retrieval_agent_workflow.c.workflow_id == workflow_id)
            .where(retrieval_agent_workflow.c.is_deleted.is_(True))
        )
        await self.database.execute(statement)

    async def set_workflow_rules(
        self, agent_id: str, workflow_id: str, account: str, rules: Rules
    ):
        await self.ensure_workflow_active(account, agent_id, workflow_id)
        statement = (
            retrieval_agent_workflow.update()
            .values(rules=rules.model_dump())
            .where(retrieval_agent_workflow.c.account == account)
            .where(retrieval_agent_workflow.c.agent_id == agent_id)
            .where(retrieval_agent_workflow.c.workflow_id == workflow_id)
            .where(self._active_workflow_condition())
        )

        await self.database.execute(statement)

    async def workflows_list(self, account: str, agent_id: str) -> List[WorkflowData]:
        statement = (
            retrieval_agent_workflow.select()
            .where(retrieval_agent_workflow.c.account == account)
            .where(retrieval_agent_workflow.c.agent_id == agent_id)
            .where(self._active_workflow_condition())
        )
        results = await self.database.fetch_all(statement)
        workflows = []
        for result in results:
            workflows.append(WorkflowData(id=result["workflow_id"], **result))  # type: ignore
        return workflows

    async def get_agent_config_basic(
        self, account: str, agent_id: str
    ) -> RetrievalAgent:
        """Loads the basic configuration without preprocess, context, generation or postprocess

        Args:
            account (str): Account ID
            agent_id (str): Agent ID
        """

        statement = (
            retrieval_agent_config.select()
            .where(retrieval_agent_config.c.account == account)
            .where(retrieval_agent_config.c.agent_id == agent_id)
        )

        result = await self.database.fetch_one(statement)

        if result is None:
            raise exceptions.NotFoundError("Agent config not found")

        return RetrievalAgent(
            account=result["account"],
            agent_id=result["agent_id"],
            description=result["description"],
            memory=result["memory"],
            title=result["title"],
            instructions=result["instructions"],
            created=result["created"],
            modified=result["modified"],
        )

    async def get_agent_config(
        self,
        account: str,
        agent_id: str,
        internal_nucliadb_url: str | None = None,
        default_memory: bool = False,
        workflow_id: str = "default",
    ) -> RetrievalAgentConfig:
        """Loads the configuration in a single query to ensure we get a consistent view

        Args:
            account (str): Account ID
            agent_id (str): Agent ID
            internal_nucliadb_url (str | None): Internal NucliaDB URL to use if no memory is set
            default_memory (bool): Whether to ignore the stored memory config and use a default one
            workflow (str): Workflow name to load the agent for
        """

        # Queries for each agent type
        queries = [
            sa.select(
                sa.literal_column(f"'{table.name}'").label("kind"),
                table.c.id,
                sa.null().label("identifier"),
                sa.null().label("name"),
                sa.null().label("provider"),
                column.label("config"),
            )
            .where(table.c.account == account)
            .where(table.c.agent_id == agent_id)
            .where(table.c.workflow_id == workflow_id)
            for (table, column) in [
                (retrieval_agent_preprocess, retrieval_agent_preprocess.c.preprocess),
                (retrieval_agent_context, retrieval_agent_context.c.context),
                (retrieval_agent_generation, retrieval_agent_generation.c.generation),
                (
                    retrieval_agent_postprocess,
                    retrieval_agent_postprocess.c.postprocess,
                ),
            ]
        ]
        # Query for drivers
        queries.append(
            sa.select(
                sa.literal_column("'driver'").label("kind"),
                retrieval_agents_drivers.c.id,
                retrieval_agents_drivers.c.identifier,
                retrieval_agents_drivers.c.driver.label("name"),
                retrieval_agents_drivers.c.provider,
                retrieval_agents_drivers.c.config,
            )
            .where(retrieval_agents_drivers.c.account == account)
            .where(retrieval_agents_drivers.c.agent_id == agent_id)
        )
        # Query for rules
        queries.append(
            sa.select(
                sa.literal_column("'rules'").label("kind"),
                sa.null().label("id"),
                sa.null().label("identifier"),
                sa.null().label("name"),
                sa.null().label("provider"),
                retrieval_agent_config.c.rules.label("config"),
            )
            .where(retrieval_agent_config.c.account == account)
            .where(retrieval_agent_config.c.agent_id == agent_id)
        )
        queries.append(
            sa.select(
                sa.literal_column("'memory'").label("kind"),
                sa.null().label("id"),
                sa.null().label("identifier"),
                sa.null().label("name"),
                sa.null().label("provider"),
                retrieval_agent_config.c.memory.label("config"),
            )
            .where(retrieval_agent_config.c.account == account)
            .where(retrieval_agent_config.c.agent_id == agent_id)
        )
        workflow_query = (
            sa.select(
                retrieval_agent_workflow.c.name,
                retrieval_agent_workflow.c.description,
                retrieval_agent_workflow.c.parameters,
                retrieval_agent_workflow.c.rules,
            )
            .where(retrieval_agent_workflow.c.account == account)
            .where(retrieval_agent_workflow.c.agent_id == agent_id)
            .where(retrieval_agent_workflow.c.workflow_id == workflow_id)
            .where(self._active_workflow_condition())
        )

        preprocess: list[Any] = []
        context: list[Any] = []
        generation: list[Any] = []
        postprocess: list[Any] = []
        drivers: list[Any] = []
        rules: Rules | None = None
        memory: MemoryConfig | None = None if not default_memory else MemoryConfig()
        rows = await self.database.fetch_all(sa.union(*queries))
        workflow_data = await self.database.fetch_one(workflow_query)
        for row in rows:
            match row["kind"]:
                case retrieval_agent_preprocess.name:
                    preprocess.append(
                        get_agent_config_instance(
                            agent_config={"id": row["id"], **row["config"]},
                            agent_type="preprocess",
                        )
                    )  # Validate config
                case retrieval_agent_context.name:
                    context.append(
                        get_agent_config_instance(
                            agent_config={"id": row["id"], **row["config"]},
                            agent_type="context",
                        )
                    )  # Validate config
                case retrieval_agent_generation.name:
                    generation.append(
                        get_agent_config_instance(
                            {"id": row["id"], **row["config"]}, agent_type="generation"
                        )
                    )  # Validate config
                case retrieval_agent_postprocess.name:
                    postprocess.append(
                        get_agent_config_instance(
                            agent_config={"id": row["id"], **row["config"]},
                            agent_type="postprocess",
                        )
                    )  # Validate config
                case "driver":
                    driver: DriverConfig = get_driver_config_instance(
                        {
                            "id": str(row["id"]),
                            "identifier": row["identifier"],
                            "name": row["name"],
                            "provider": row["provider"],
                            "config": row["config"],
                        }
                    )  # Validate config
                    decrypt_fields(driver.config)
                    drivers.append(driver)

                case "rules":
                    rules = Rules.model_validate(row["config"])
                case "memory":
                    if not default_memory and row["config"] is not None:
                        memory = MemoryConfig.model_validate(row["config"])

        if workflow_data is None:
            raise exceptions.NotFoundError("Workflow not found")

        workflow = WorkflowData(
            id=workflow_id,
            name=workflow_data["name"],
            description=workflow_data["description"],
            parameters=workflow_data["parameters"],
            rules=Rules.model_validate(workflow_data["rules"]),
            required=workflow_data["required"] if "required" in workflow_data else [],  # noqa
        )

        if rules is None:
            raise exceptions.NotFoundError("Agent config not found")

        if memory is None and internal_nucliadb_url is not None and not default_memory:
            memory = MemoryConfig(
                nucliadb=NucliaDBMemoryConfig(
                    url=internal_nucliadb_url, kbid=agent_id, internal=True
                )
            )

        if memory is None:
            raise Exception("Agent memory config not found")

        return RetrievalAgentConfig(
            preprocess=preprocess,
            context=context,
            generation=generation,
            postprocess=postprocess,
            drivers=drivers,
            rules=rules,
            memory=memory,
            workflow=workflow,
        )

    async def set_rules(self, agent_id: str, account: str, rules: Rules):
        statement = (
            retrieval_agent_config.update()
            .values(rules=rules.model_dump())
            .where(retrieval_agent_config.c.account == account)
            .where(retrieval_agent_config.c.agent_id == agent_id)
        )

        await self.database.execute(statement)

    async def add_prompt(
        self, agent_id: str, account: str, prompt: PromptConfig
    ) -> str:
        statement = retrieval_agent_prompts.insert().values(
            account=account,
            agent_id=agent_id,
            name=prompt.name,
            prompt=prompt.prompt,
            description=prompt.description,
            arguments=[x.model_dump() for x in prompt.arguments]
            if prompt.arguments is not None
            else None,
            icons=prompt.icons,
            meta=prompt.meta,
        )
        result = await self.database.execute(statement)
        return str(result)

    async def set_prompt(
        self, agent_id: str, account: str, prompt_id: str, prompt: PromptConfig
    ):
        statement = (
            retrieval_agent_prompts.update()
            .values(
                name=prompt.name,
                description=prompt.description,
                prompt=prompt.prompt,
                arguments=[x.model_dump() for x in prompt.arguments]
                if prompt.arguments is not None
                else None,
                icons=prompt.icons,
                meta=prompt.meta,
            )
            .where(retrieval_agent_prompts.c.account == account)
            .where(retrieval_agent_prompts.c.agent_id == agent_id)
            .where(retrieval_agent_prompts.c.id == prompt_id)
        )

        await self.database.execute(statement)

    async def delete_prompt(self, agent_id: str, account: str, prompt_id: str):
        statement = (
            retrieval_agent_prompts.delete()
            .where(retrieval_agent_prompts.c.account == account)
            .where(retrieval_agent_prompts.c.agent_id == agent_id)
            .where(retrieval_agent_prompts.c.id == prompt_id)
        )

        await self.database.execute(statement)

    async def get_prompt(
        self, agent_id: str, account: str, prompt_id: str
    ) -> PromptConfig:
        statement = (
            retrieval_agent_prompts.select()
            .where(retrieval_agent_prompts.c.account == account)
            .where(retrieval_agent_prompts.c.agent_id == agent_id)
            .where(retrieval_agent_prompts.c.id == prompt_id)
        )
        result = await self.database.fetch_one(statement)
        if result is None:
            raise exceptions.NotFoundError("Prompt not found")

        prompt = PromptConfig(
            name=result["name"],
            description=result["description"],
            prompt=result["prompt"],
            arguments=[
                PromptArgument.model_validate(arg) for arg in result["arguments"]
            ]
            if result["arguments"] is not None
            else None,
            icons=result["icons"],
            meta=result["meta"],
            prompt_id=str(result["id"]),
        )

        return prompt

    async def get_prompts(self, agent_id: str, account: str) -> List[PromptConfig]:
        statement = (
            retrieval_agent_prompts.select()
            .where(retrieval_agent_config.c.account == account)
            .where(retrieval_agent_config.c.agent_id == agent_id)
        )
        result = await self.database.fetch_all(statement)
        prompts = []
        for row in result:
            prompt = PromptConfig(
                name=row["name"],
                description=row["description"],
                prompt=row["prompt"],
                arguments=[
                    PromptArgument.model_validate(arg) for arg in row["arguments"]
                ]
                if row["arguments"] is not None
                else None,
                icons=row["icons"],
                meta=row["meta"],
                prompt_id=str(row["id"]),
            )
            prompts.append(prompt)

        return prompts

    async def set_memory(self, agent_id: str, account: str, memory: MemoryConfig):
        statement = (
            retrieval_agent_config.update()
            .values(memory=memory.model_dump())
            .where(retrieval_agent_config.c.account == account)
            .where(retrieval_agent_config.c.agent_id == agent_id)
        )

        await self.database.execute(statement)

    async def get_rules(self, account: str, agent_id: str) -> Rules:
        statement = (
            retrieval_agent_config.select()
            .where(retrieval_agent_config.c.account == account)
            .where(retrieval_agent_config.c.agent_id == agent_id)
        )
        result = await self.database.fetch_one(statement)
        rules = Rules(rules=[])

        if result is not None:
            rules = Rules.model_validate(result["rules"])

        return rules

    async def get_workflow_rules(
        self, account: str, agent_id: str, workflow_id: str
    ) -> Rules:
        statement = (
            retrieval_agent_workflow.select()
            .where(retrieval_agent_workflow.c.account == account)
            .where(retrieval_agent_workflow.c.agent_id == agent_id)
            .where(retrieval_agent_workflow.c.workflow_id == workflow_id)
            .where(self._active_workflow_condition())
        )
        result = await self.database.fetch_one(statement)
        if result is None:
            raise exceptions.NotFoundError("Workflow not found")

        return Rules.model_validate(result["rules"])

    async def add_preprocess(
        self,
        agent_id: str,
        account: str,
        agent: BaseModel,
        workflow_id: str = "default",
    ):
        await self.ensure_workflow_active(account, agent_id, workflow_id)
        statement = retrieval_agent_preprocess.insert().values(
            account=account,
            agent_id=agent_id,
            workflow_id=workflow_id,
            preprocess=agent.model_dump(),
        )
        result = await self.database.execute(statement)
        return str(result)

    async def patch_preprocess(
        self,
        agent_id: str,
        account: str,
        preprocess: str,
        agent: BaseModel,
        workflow_id: str = "default",
    ):
        await self.ensure_workflow_active(account, agent_id, workflow_id)
        statement = (
            retrieval_agent_preprocess.update()
            .where(retrieval_agent_preprocess.c.id == preprocess)
            .where(retrieval_agent_preprocess.c.agent_id == agent_id)
            .where(retrieval_agent_preprocess.c.account == account)
            .where(retrieval_agent_preprocess.c.workflow_id == workflow_id)
            .values(preprocess=agent.model_dump())
        )
        await self.database.execute(statement)

    async def delete_preprocess(
        self, account: str, agent_id: str, preprocess: str, workflow_id: str = "default"
    ):
        await self.ensure_workflow_active(account, agent_id, workflow_id)
        statement = (
            retrieval_agent_preprocess.delete()
            .where(retrieval_agent_preprocess.c.account == account)
            .where(retrieval_agent_preprocess.c.agent_id == agent_id)
            .where(retrieval_agent_preprocess.c.workflow_id == workflow_id)
            .where(retrieval_agent_preprocess.c.id == preprocess)
        )
        await self.database.execute(statement)

    async def get_preprocess(
        self, account: str, agent_id: str, workflow_id: str = "default"
    ) -> List[AgentConfig]:
        await self.ensure_workflow_active(account, agent_id, workflow_id)
        statement = (
            retrieval_agent_preprocess.select()
            .where(retrieval_agent_preprocess.c.account == account)
            .where(retrieval_agent_preprocess.c.agent_id == agent_id)
            .where(retrieval_agent_preprocess.c.workflow_id == workflow_id)
        )
        results = await self.database.fetch_all(statement)
        preprocess = []
        for result in results:
            base_config = get_agent_config_instance(
                agent_config=result["preprocess"], agent_type="preprocess"
            )
            base_config.id = str(result["id"])
            preprocess.append(base_config)

        return preprocess

    async def delete_postprocess(
        self,
        account: str,
        agent_id: str,
        postprocess: str,
        workflow_id: str = "default",
    ):
        await self.ensure_workflow_active(account, agent_id, workflow_id)
        statement = (
            retrieval_agent_postprocess.delete()
            .where(retrieval_agent_postprocess.c.account == account)
            .where(retrieval_agent_postprocess.c.agent_id == agent_id)
            .where(retrieval_agent_postprocess.c.workflow_id == workflow_id)
            .where(retrieval_agent_postprocess.c.id == postprocess)
        )
        await self.database.execute(statement)

    async def add_postprocess(
        self,
        agent_id: str,
        account: str,
        agent: BaseModel,
        workflow_id: str = "default",
    ):
        await self.ensure_workflow_active(account, agent_id, workflow_id)
        statement = retrieval_agent_postprocess.insert().values(
            account=account,
            agent_id=agent_id,
            postprocess=agent.model_dump(),
            workflow_id=workflow_id,
        )
        result = await self.database.execute(statement)
        return str(result)

    async def patch_postprocess(
        self,
        agent_id: str,
        account: str,
        postprocess: str,
        agent: BaseModel,
        workflow_id: str = "default",
    ):
        await self.ensure_workflow_active(account, agent_id, workflow_id)
        statement = (
            retrieval_agent_postprocess.update()
            .where(retrieval_agent_postprocess.c.id == postprocess)
            .where(retrieval_agent_postprocess.c.agent_id == agent_id)
            .where(retrieval_agent_postprocess.c.account == account)
            .where(retrieval_agent_postprocess.c.workflow_id == workflow_id)
            .values(postprocess=agent.model_dump())
        )
        await self.database.execute(statement)

    async def get_postprocess(
        self, account: str, agent_id: str, workflow_id: str = "default"
    ) -> List[AgentConfig]:
        await self.ensure_workflow_active(account, agent_id, workflow_id)
        statement = (
            retrieval_agent_postprocess.select()
            .where(retrieval_agent_postprocess.c.account == account)
            .where(retrieval_agent_postprocess.c.agent_id == agent_id)
            .where(retrieval_agent_postprocess.c.workflow_id == workflow_id)
        )
        results = await self.database.fetch_all(statement)
        postprocess = []
        for result in results:
            base_config = get_agent_config_instance(
                agent_config=result["postprocess"], agent_type="postprocess"
            )
            base_config.id = str(result["id"])
            postprocess.append(base_config)

        return postprocess

    async def add_context(
        self,
        agent_id: str,
        account: str,
        agent: BaseModel,
        workflow_id: str = "default",
    ):
        await self.ensure_workflow_active(account, agent_id, workflow_id)
        statement = retrieval_agent_context.insert().values(
            account=account,
            agent_id=agent_id,
            context=agent.model_dump(),
            workflow_id=workflow_id,
        )
        result = await self.database.execute(statement)

        return str(result)

    async def delete_context(
        self, agent_id: str, account: str, context: UUID, workflow_id: str = "default"
    ):
        await self.ensure_workflow_active(account, agent_id, workflow_id)
        statement = (
            retrieval_agent_context.delete()
            .where(retrieval_agent_context.c.id == context)
            .where(retrieval_agent_context.c.agent_id == agent_id)
            .where(retrieval_agent_context.c.account == account)
            .where(retrieval_agent_context.c.workflow_id == workflow_id)
        )
        await self.database.execute(statement)

    async def patch_context(
        self,
        agent_id: str,
        account: str,
        context: UUID,
        agent: BaseModel,
        workflow_id: str = "default",
    ):
        await self.ensure_workflow_active(account, agent_id, workflow_id)
        statement = (
            retrieval_agent_context.update()
            .where(retrieval_agent_context.c.id == context)
            .where(retrieval_agent_context.c.agent_id == agent_id)
            .where(retrieval_agent_context.c.account == account)
            .where(retrieval_agent_context.c.workflow_id == workflow_id)
            .values(context=agent.model_dump())
        )
        await self.database.execute(statement)

    async def get_context(
        self, account: str, agent_id: str, workflow_id: str = "default"
    ) -> List[AgentConfig]:
        await self.ensure_workflow_active(account, agent_id, workflow_id)
        statement = (
            retrieval_agent_context.select()
            .where(retrieval_agent_context.c.account == account)
            .where(retrieval_agent_context.c.agent_id == agent_id)
            .where(retrieval_agent_context.c.workflow_id == workflow_id)
        )
        results = await self.database.fetch_all(statement)
        context = []
        for result in results:
            base_config = get_agent_config_instance(
                agent_config=result["context"], agent_type="context"
            )
            base_config.id = str(result["id"])
            context.append(base_config)

        return context

    async def add_generation(
        self,
        agent_id: str,
        account: str,
        agent: BaseModel,
        workflow_id: str = "default",
    ):
        await self.ensure_workflow_active(account, agent_id, workflow_id)
        statement = retrieval_agent_generation.insert().values(
            account=account,
            agent_id=agent_id,
            generation=agent.model_dump(),
            workflow_id=workflow_id,
        )
        result = await self.database.execute(statement)

        return str(result)

    async def delete_generation(
        self,
        agent_id: str,
        account: str,
        generation: str,
        workflow_id: str = "default",
    ):
        await self.ensure_workflow_active(account, agent_id, workflow_id)
        statement = (
            retrieval_agent_generation.delete()
            .where(retrieval_agent_generation.c.id == generation)
            .where(retrieval_agent_generation.c.agent_id == agent_id)
            .where(retrieval_agent_generation.c.account == account)
            .where(retrieval_agent_generation.c.workflow_id == workflow_id)
        )
        await self.database.execute(statement)

    async def patch_generation(
        self,
        agent_id: str,
        account: str,
        generation: str,
        agent: BaseModel,
        workflow_id: str = "default",
    ):
        await self.ensure_workflow_active(account, agent_id, workflow_id)
        statement = (
            retrieval_agent_generation.update()
            .where(retrieval_agent_generation.c.id == generation)
            .where(retrieval_agent_generation.c.agent_id == agent_id)
            .where(retrieval_agent_generation.c.account == account)
            .where(retrieval_agent_generation.c.workflow_id == workflow_id)
            .values(generation=agent.model_dump())
        )
        await self.database.execute(statement)

    async def get_generation(
        self, account: str, agent_id: str, workflow_id: str = "default"
    ) -> List[AgentConfig]:
        await self.ensure_workflow_active(account, agent_id, workflow_id)
        statement = (
            retrieval_agent_generation.select()
            .where(retrieval_agent_generation.c.account == account)
            .where(retrieval_agent_generation.c.agent_id == agent_id)
            .where(retrieval_agent_generation.c.workflow_id == workflow_id)
        )
        results = await self.database.fetch_all(statement)
        generation = []
        for result in results:
            base_config = get_agent_config_instance(
                agent_config=result["generation"], agent_type="generation"
            )
            base_config.id = str(result["id"])
            generation.append(base_config)

        return generation

    async def export(
        self, account: str, agent_id: str, passphrase: str
    ) -> tuple[bytes, str | None]:
        # Passphrase validation
        if len(passphrase) < 16:
            raise Exception("Passphrase too short, minimum 16 characters required")
        workflow_config = {}
        default_config = None
        workflows = [wf.id for wf in await self.workflows_list(account, agent_id)]
        for workflow in workflows:
            try:
                agent_config = await self.get_agent_config(
                    account, agent_id, default_memory=True, workflow_id=workflow
                )
                if workflow == "default":
                    default_config = agent_config
                workflow_config[workflow] = agent_config
            except Exception:
                logger.exception("Retrieval agent not found for export")
                raise Exception("Retrieval agent not found for export")

        # XXX: ID sanitization not required since current implementation ignores IDs on import

        export_model = RetrievalAgentExportV1(
            agent_config=default_config,
            agent_config_workflows=workflow_config,
            prompts=await self.get_prompts(agent_id, account),
        )
        export_bytes = export_model.model_dump_json().encode("utf-8")
        key, salt = fernet_key_from_passphrase(passphrase, None)
        fernet = Fernet(key)
        try:
            encrypted_bytes = salt + fernet.encrypt(export_bytes)
        except Exception:
            logger.exception("Error encrypting retrieval agent export")
            raise Exception("Error encrypting export")

        return encrypted_bytes, None

    async def import_config(
        self,
        account: str,
        agent_id: str,
        import_file: UploadFile,
        passphrase: str,
        overwrite: bool,
    ):
        # Get the current agent config
        try:
            destination_agent_config = await self.get_agent_config(
                account, agent_id, default_memory=True, workflow_id="default"
            )
        except exceptions.NotFoundError:
            raise
        # If not overwriting and the agent config is not empty, raise error
        if not destination_agent_config.is_empty():
            if not overwrite:
                raise exceptions.InvalidTargetAgentError()
            else:
                # Delete current configuration
                await self.delete_agent(account=account, agent_id=agent_id)
                # Recreate empty configuration
                await self.add_agent(
                    account=account,
                    agent_id=agent_id,
                    memory=MemoryConfig(),
                    rules=Rules(rules=[]),
                )

        # Read the salt (first 16 bytes)
        salt = await import_file.read(16)
        key, _ = fernet_key_from_passphrase(passphrase, salt)
        fernet = Fernet(key)
        # Read the rest of the file in chunks

        import_bytes = bytearray()
        while True:
            chunk = await import_file.read(self.settings.export_read_chunk_size)
            if not chunk:
                break
            import_bytes.extend(chunk)
            if len(import_bytes) > self.settings.export_read_max_size:
                await import_file.close()
                raise exceptions.ParseExportError("Import file too large")
        await import_file.close()
        # Decrypt the bytes
        try:
            encrypted_bytes = fernet.decrypt(bytes(import_bytes))
        except Exception as e:
            raise exceptions.ExportEncryptionError from e
        # Load the model
        try:
            parsed_export = retrievalAgentAdapter.validate_json(
                encrypted_bytes.decode("utf-8")
            )
        except ValidationError as e:
            raise exceptions.ParseExportError from e

        if isinstance(parsed_export, RetrievalAgentExportV1):
            workflow_configs: dict[str, RetrievalAgentConfig] = {}
            agent_config = parsed_export.agent_config
            if agent_config is not None:
                workflow_id = "default"
                workflow_configs[workflow_id] = agent_config
            for wf_id, wf_config in parsed_export.agent_config_workflows.items():
                workflow_configs[wf_id] = wf_config

            drivers = {}
            agent_rules = None
            try:
                for workflow_id, agent_config_workflow in workflow_configs.items():
                    for driver in agent_config_workflow.drivers:
                        drivers[driver.name] = driver
                    agent_rules = agent_config_workflow.rules
                    wf = agent_config_workflow.workflow
                    if workflow_id == "default":
                        await self.set_workflow(
                            account=account,
                            agent_id=agent_id,
                            workflow_id=workflow_id,
                            item=WorkflowUpdate(
                                name=wf.name,
                                description=wf.description or "",
                                parameters=wf.parameters or {},
                                rules=wf.rules,
                            ),
                        )
                    else:
                        await self.add_workflow(
                            account=account,
                            agent_id=agent_id,
                            item=WorkflowInput(
                                id=workflow_id,
                                name=wf.name,
                                description=wf.description,
                                parameters=wf.parameters,
                                rules=wf.rules or Rules(rules=[]),
                            ),
                        )

                    # Store preprocess
                    for preprocess in agent_config_workflow.preprocess:
                        await self.add_preprocess(
                            agent_id=agent_id,
                            account=account,
                            agent=preprocess,
                            workflow_id=workflow_id,
                        )
                    # Store context
                    for context in agent_config_workflow.context:
                        await self.add_context(
                            agent_id=agent_id,
                            account=account,
                            agent=context,
                            workflow_id=workflow_id,
                        )
                    # Store generation
                    for generation in agent_config_workflow.generation:
                        await self.add_generation(
                            agent_id=agent_id,
                            account=account,
                            agent=generation,
                            workflow_id=workflow_id,
                        )
                    # Store postprocess
                    for postprocess in agent_config_workflow.postprocess:
                        await self.add_postprocess(
                            agent_id=agent_id,
                            account=account,
                            agent=postprocess,
                            workflow_id=workflow_id,
                        )

                if agent_rules is not None:
                    # Store rules
                    await self.set_rules(
                        agent_id=agent_id,
                        account=account,
                        rules=agent_rules,
                    )
                # XXX: We are pruposely not importing memory configuration and will use the one already set
                # As KBs are created with the agent, if when importing we overwrite, we leave a dangling KB
                # And cross region imports with KB memory would break as well
                # This can be revisited in future versions if needed

                # Store drivers — done once after all workflows are processed to avoid
                # duplicate insertions (drivers are global per-agent but appear in each
                # workflow's exported config).
                for driver in drivers.values():
                    await self.add_driver(
                        agent_id=agent_id, account=account, config=driver
                    )
                for prompt in parsed_export.prompts:
                    await self.add_prompt(
                        agent_id=agent_id,
                        account=account,
                        prompt=prompt,
                    )
            except Exception as e:
                raise exceptions.ParseExportError(
                    "Failed to import retrieval agent configuration"
                ) from e
        else:
            raise exceptions.ParseExportError("Unsupported export version")


def update_driver_config(
    config: DriverConfig, previous_config: DriverConfig
) -> DriverConfig:
    # Generate configuration for values set in the request. All values present on
    # the request will override values of the current configuration. If there was
    # any value on the stored configuration NOT SET yet, and it's not on the update
    # the default will be added.
    desired_configuration = config.model_dump(exclude_unset=True)
    updated_configuration = previous_config.model_dump()

    return config.__class__(**deep_update(updated_configuration, desired_configuration))


def deep_update(original: dict, updates: dict) -> dict:
    """
    Recursively update a nested dictionary.
    """
    for key, value in updates.items():
        if (
            key in original
            and isinstance(original[key], dict)
            and isinstance(value, dict)
        ):
            deep_update(original[key], value)
        else:
            original[key] = value
    return original
