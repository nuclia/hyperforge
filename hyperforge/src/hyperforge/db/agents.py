import datetime
import json
from typing import Any, List
from uuid import UUID, uuid4

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
    GLOBAL_REGISTRY,
    get_agent_config_instance,
    get_driver_config_klass,
    get_toolkit_config_instance,
)
from hyperforge.database import metadata
from hyperforge.db import exceptions, logger
from hyperforge.db.encryption import (
    decrypt_data,
    decrypt_fields,
    encrypt_data,
    encrypt_fields,
    fernet_key_from_passphrase,
)
from hyperforge.db.settings import DataManagerSettings
from hyperforge.db.workflow_migration import transpile_stages_to_harness
from hyperforge.driver import DriverConfig
from hyperforge.harness.config import (
    HarnessAgentConfig,
    HarnessAgentExportV2,
    _export_union_adapter,
)
from hyperforge.harness_sdk import SYSTEM_PROMPT, HarnessSkill, ToolkitConfig
from hyperforge.llm_config import llm_defaults
from hyperforge.models import MemoryConfig, NucliaDBMemoryConfig, Rules
from hyperforge.prompts import PromptArgument, PromptConfig
from hyperforge.retrieval.config import RetrievalAgentConfig, RetrievalAgentExportV1
from hyperforge.workflows import (
    RetrievalAgent,
    WorkflowData,
    WorkflowHarnessUpdate,
    WorkflowInput,
    WorkflowUpdate,
)

SERVICE_NAME = "TASK_MANAGER"
EXPIRATION = 7 * 24
WORKFLOW_PURGE_RETENTION = datetime.timedelta(days=15)


def utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _decode_json_col(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


retrieval_agent_workflow = sa.Table(
    "retrieval_agent_workflow",
    metadata,
    sa.Column("account", sa.String, primary_key=True, nullable=False, index=True),
    sa.Column("agent_id", sa.String, primary_key=True, nullable=False, index=True),
    sa.Column("workflow_id", sa.String, primary_key=True, nullable=False, index=True),
    sa.Column("name", sa.String, nullable=False),
    sa.Column("description", sa.String, nullable=True),
    sa.Column("parameters", JSONB, nullable=True),
    sa.Column("required", JSONB, nullable=True),
    sa.Column("rules", JSONB, nullable=True),
    sa.Column("execution_mode", sa.String, nullable=False, default="loop"),
    sa.Column("harness_config", JSONB, nullable=False, default=dict),
    sa.Column("workflow_code", sa.Text, nullable=True),
    sa.Column("legacy_stages_snapshot", JSONB, nullable=True),
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

agent_workflow_toolkits = sa.Table(
    "agent_workflow_toolkits",
    metadata,
    sa.Column(
        "id",
        pg_dialect.UUID,
        primary_key=True,
        server_default=sa.func.uuid_generate_v4(),
    ),
    sa.Column("account", sa.String, nullable=False, index=True),
    sa.Column("agent_id", sa.String, nullable=False, index=True),
    sa.Column("workflow_id", sa.String, nullable=False, index=True),
    sa.Column("namespace", sa.String, nullable=False),
    sa.Column("toolkit", sa.String, nullable=False),
    sa.Column("title", sa.String, nullable=True),
    sa.Column("lazy_load", sa.Boolean, nullable=False, default=False),
    sa.Column("ingest_skills", sa.Boolean, nullable=False, default=True),
    sa.Column("config", JSONB, nullable=False, default=dict),
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
    sa.UniqueConstraint(
        "account",
        "agent_id",
        "workflow_id",
        "namespace",
        name="uq_agent_workflow_toolkits_namespace",
    ),
)

agent_skills = sa.Table(
    "agent_skills",
    metadata,
    sa.Column(
        "id",
        pg_dialect.UUID,
        primary_key=True,
        server_default=sa.func.uuid_generate_v4(),
    ),
    sa.Column("account", sa.String, nullable=False, index=True),
    sa.Column("agent_id", sa.String, nullable=False, index=True),
    sa.Column("skill_id", sa.String, nullable=False),
    sa.Column("name", sa.String, nullable=False),
    sa.Column("description", sa.String, nullable=False),
    sa.Column("instructions", sa.Text, nullable=False),
    sa.Column("recommended_tools", JSONB, nullable=False, default=list),
    sa.Column("code_examples", JSONB, nullable=False, default=list),
    sa.Column("preload", sa.Boolean, nullable=False, default=False),
    sa.Column("meta", JSONB, nullable=True),
    sa.Column("created", sa.DateTime, default=sa.func.now()),
    sa.Column("modified", sa.DateTime, onupdate=sa.func.now()),
    sa.ForeignKeyConstraint(
        ["account", "agent_id"],
        ["retrieval_agent_config.account", "retrieval_agent_config.agent_id"],
        onupdate="CASCADE",
        ondelete="CASCADE",
    ),
    sa.UniqueConstraint(
        "account", "agent_id", "skill_id", name="uq_agent_skills_skill_id"
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
    sa.Column("account", sa.String, nullable=False, index=True),
    sa.Column("agent_id", sa.String, nullable=False, index=True),
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
    sa.Column("agent_id", sa.String, primary_key=True, nullable=False, index=True),
    sa.Column("rules", JSONB, nullable=False),
    sa.Column("created", sa.DateTime, default=sa.func.now()),
    sa.Column("modified", sa.DateTime, onupdate=sa.func.now()),
    sa.Column("memory", JSONB, nullable=False),
    sa.Column("description", sa.String, nullable=True),
    sa.Column("title", sa.String, nullable=True),
    sa.Column("instructions", sa.String, nullable=True),
)

sync_oauth_credentials = sa.Table(
    "sync_oauth_credentials",
    metadata,
    sa.Column("account", sa.String, primary_key=True, nullable=False),
    sa.Column("user_id", sa.String, primary_key=True, nullable=False),
    sa.Column("agent_id", sa.String, primary_key=True, nullable=False),
    sa.Column("provider", sa.String, primary_key=True, nullable=False),
    sa.Column("sync_config_id", sa.String, primary_key=True, nullable=False),
    sa.Column("encrypted_credentials", sa.Text, nullable=False),
    sa.Column(
        "created", sa.DateTime(timezone=True), nullable=False, default=sa.func.now()
    ),
    sa.Column(
        "modified",
        sa.DateTime(timezone=True),
        nullable=False,
        default=sa.func.now(),
        onupdate=sa.func.now(),
    ),
    sa.ForeignKeyConstraint(
        ["account", "agent_id"],
        ["retrieval_agent_config.account", "retrieval_agent_config.agent_id"],
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


def _split_toolkit_row(
    config: ToolkitConfig,
) -> tuple[str, str, str | None, bool, bool, dict[str, Any]]:
    dumped = config.model_dump()
    dumped.pop("id", None)
    provider = str(dumped.pop("toolkit"))
    namespace = str(dumped.pop("namespace", None) or provider)
    title = dumped.pop("title", None)
    lazy_load = bool(dumped.pop("lazy_load", False))
    ingest_skills = bool(dumped.pop("ingest_skills", True))
    return namespace, provider, title, lazy_load, ingest_skills, dumped


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

    async def get_sync_oauth_credentials(
        self,
        *,
        account: str,
        user_id: str,
        agent_id: str,
        provider: str,
        sync_config_id: str,
    ) -> dict[str, str] | None:
        statement = sa.select(sync_oauth_credentials.c.encrypted_credentials).where(
            sync_oauth_credentials.c.account == account,
            sync_oauth_credentials.c.user_id == user_id,
            sync_oauth_credentials.c.agent_id == agent_id,
            sync_oauth_credentials.c.provider == provider,
            sync_oauth_credentials.c.sync_config_id == sync_config_id,
        )
        row = await self.database.fetch_one(statement)
        if row is None:
            return None

        try:
            credentials = json.loads(decrypt_data(row["encrypted_credentials"]))
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid stored Sync OAuth credentials") from exc
        if not isinstance(credentials, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in credentials.items()
        ):
            raise ValueError("Invalid stored Sync OAuth credentials")
        return credentials

    async def upsert_sync_oauth_credentials(
        self,
        *,
        account: str,
        user_id: str,
        agent_id: str,
        provider: str,
        sync_config_id: str,
        credentials: dict[str, str],
    ) -> None:
        encrypted_credentials = encrypt_data(
            json.dumps(credentials, sort_keys=True, separators=(",", ":"))
        )
        statement = pg_dialect.insert(sync_oauth_credentials).values(
            account=account,
            user_id=user_id,
            agent_id=agent_id,
            provider=provider,
            sync_config_id=sync_config_id,
            encrypted_credentials=encrypted_credentials,
        )
        statement = statement.on_conflict_do_update(
            index_elements=[
                sync_oauth_credentials.c.account,
                sync_oauth_credentials.c.user_id,
                sync_oauth_credentials.c.agent_id,
                sync_oauth_credentials.c.provider,
                sync_oauth_credentials.c.sync_config_id,
            ],
            set_={
                "encrypted_credentials": encrypted_credentials,
                "modified": sa.func.now(),
            },
        )
        await self.database.execute(statement)

    async def upsert_sync_oauth_credentials_batch(
        self,
        *,
        account: str,
        user_id: str,
        agent_id: str,
        credentials_by_config: dict[str, tuple[str, dict[str, str]]],
    ) -> None:
        async with self.database.transaction():
            for sync_config_id, (
                provider,
                credentials,
            ) in credentials_by_config.items():
                await self.upsert_sync_oauth_credentials(
                    account=account,
                    user_id=user_id,
                    agent_id=agent_id,
                    provider=provider,
                    sync_config_id=sync_config_id,
                    credentials=credentials,
                )

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
            return
        updated_config: DriverConfig = update_driver_config(config, previous_config)
        statement = (
            retrieval_agents_drivers.update()
            .where(retrieval_agents_drivers.c.account == account)
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
                "config": _decode_json_col(result["config"], {}),
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
                    "config": _decode_json_col(result["config"], {}),
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
            execution_mode="loop",
            harness_config={"model": llm_defaults.smart},
            workflow_code=None,
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
        harness_cfg = dict(item.harness_config)
        if item.execution_mode == "loop" and not harness_cfg.get("model"):
            harness_cfg["model"] = llm_defaults.smart
        statement = retrieval_agent_workflow.insert().values(
            account=account,
            agent_id=agent_id,
            workflow_id=item.id,
            name=item.name,
            description=item.description,
            parameters=item.parameters,
            required=item.required,
            rules=(
                item.rules.model_dump() if item.rules else Rules(rules=[]).model_dump()
            ),
            execution_mode=item.execution_mode,
            harness_config=harness_cfg,
            workflow_code=item.workflow_code,
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
        values: dict[str, Any] = {
            "name": item.name,
            "description": item.description,
            "parameters": item.parameters,
            "required": item.required,
            "rules": (
                item.rules.model_dump() if item.rules else Rules(rules=[]).model_dump()
            ),
        }
        if item.execution_mode is not None:
            values["execution_mode"] = item.execution_mode
        if item.harness_config is not None:
            values["harness_config"] = item.harness_config
        if item.workflow_code is not None:
            values["workflow_code"] = item.workflow_code

        statement = (
            retrieval_agent_workflow.update()
            .where(retrieval_agent_workflow.c.account == account)
            .where(retrieval_agent_workflow.c.agent_id == agent_id)
            .where(retrieval_agent_workflow.c.workflow_id == workflow_id)
            .where(self._active_workflow_condition())
            .values(**values)
        )
        await self.database.execute(statement)

    async def set_workflow_harness(
        self,
        account: str,
        agent_id: str,
        workflow_id: str,
        item: WorkflowHarnessUpdate,
    ) -> None:
        await self.ensure_workflow_active(account, agent_id, workflow_id)
        statement = (
            retrieval_agent_workflow.update()
            .where(retrieval_agent_workflow.c.account == account)
            .where(retrieval_agent_workflow.c.agent_id == agent_id)
            .where(retrieval_agent_workflow.c.workflow_id == workflow_id)
            .where(self._active_workflow_condition())
            .values(
                execution_mode=item.execution_mode,
                harness_config=item.harness_config,
                workflow_code=item.workflow_code,
            )
        )
        await self.database.execute(statement)

    async def get_workflow_harness(
        self, account: str, agent_id: str, workflow_id: str
    ) -> WorkflowHarnessUpdate:
        statement = (
            retrieval_agent_workflow.select()
            .where(retrieval_agent_workflow.c.account == account)
            .where(retrieval_agent_workflow.c.agent_id == agent_id)
            .where(retrieval_agent_workflow.c.workflow_id == workflow_id)
            .where(self._active_workflow_condition())
        )
        row = await self.database.fetch_one(statement)
        if row is None:
            raise exceptions.NotFoundError("Workflow not found")
        return WorkflowHarnessUpdate(
            execution_mode=row["execution_mode"],
            harness_config=_decode_json_col(row["harness_config"], {}),
            workflow_code=row["workflow_code"],
        )

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
            row_dict = dict(result._mapping)
            row_dict["parameters"] = _decode_json_col(row_dict.get("parameters"), None)
            row_dict["required"] = _decode_json_col(row_dict.get("required"), []) or []
            row_dict["rules"] = _decode_json_col(row_dict.get("rules"), {"rules": []})
            row_dict["harness_config"] = _decode_json_col(
                row_dict.get("harness_config"), {}
            )
            workflows.append(WorkflowData(id=row_dict["workflow_id"], **row_dict))
        return workflows

    # ------------------------------------------------------------------
    # Workflow Toolkits CRUD
    # ------------------------------------------------------------------

    async def add_toolkit(
        self,
        *,
        account: str,
        agent_id: str,
        toolkit: ToolkitConfig,
        workflow_id: str = "default",
    ) -> str:
        await self.ensure_workflow_active(account, agent_id, workflow_id)
        namespace, provider, title, lazy_load, ingest_skills, cfg = _split_toolkit_row(
            toolkit
        )
        stmt = agent_workflow_toolkits.insert().values(
            account=account,
            agent_id=agent_id,
            workflow_id=workflow_id,
            namespace=namespace,
            toolkit=provider,
            title=title,
            lazy_load=lazy_load,
            ingest_skills=ingest_skills,
            config=cfg,
        )
        res = await self.database.execute(stmt)
        return str(res)

    async def patch_toolkit(
        self,
        *,
        account: str,
        agent_id: str,
        toolkit_id: str | UUID,
        toolkit: ToolkitConfig,
        workflow_id: str = "default",
    ) -> None:
        await self.ensure_workflow_active(account, agent_id, workflow_id)
        namespace, provider, title, lazy_load, ingest_skills, cfg = _split_toolkit_row(
            toolkit
        )
        stmt = (
            agent_workflow_toolkits.update()
            .where(
                agent_workflow_toolkits.c.id == str(toolkit_id),
                agent_workflow_toolkits.c.account == account,
                agent_workflow_toolkits.c.agent_id == agent_id,
                agent_workflow_toolkits.c.workflow_id == workflow_id,
            )
            .values(
                namespace=namespace,
                toolkit=provider,
                title=title,
                lazy_load=lazy_load,
                ingest_skills=ingest_skills,
                config=cfg,
            )
        )
        await self.database.execute(stmt)

    async def delete_toolkit(
        self,
        *,
        account: str,
        agent_id: str,
        toolkit_id: str | UUID,
        workflow_id: str = "default",
    ) -> None:
        await self.ensure_workflow_active(account, agent_id, workflow_id)
        stmt = agent_workflow_toolkits.delete().where(
            agent_workflow_toolkits.c.id == str(toolkit_id),
            agent_workflow_toolkits.c.account == account,
            agent_workflow_toolkits.c.agent_id == agent_id,
            agent_workflow_toolkits.c.workflow_id == workflow_id,
        )
        await self.database.execute(stmt)

    async def get_toolkits(
        self,
        account: str,
        agent_id: str,
        workflow_id: str = "default",
    ) -> List[ToolkitConfig]:
        await self.ensure_workflow_active(account, agent_id, workflow_id)
        stmt = (
            agent_workflow_toolkits.select()
            .where(
                agent_workflow_toolkits.c.account == account,
                agent_workflow_toolkits.c.agent_id == agent_id,
                agent_workflow_toolkits.c.workflow_id == workflow_id,
            )
            .order_by(agent_workflow_toolkits.c.created.asc())
        )
        rows = await self.database.fetch_all(stmt)
        toolkits: list[ToolkitConfig] = []
        for row in rows:
            raw_cfg = _decode_json_col(row["config"], {})
            merged = {
                **raw_cfg,
                "id": str(row["id"]),
                "toolkit": row["toolkit"],
                "namespace": row["namespace"],
                "title": row["title"],
                "lazy_load": row["lazy_load"],
                "ingest_skills": row["ingest_skills"],
            }
            try:
                toolkits.append(get_toolkit_config_instance(merged))
            except Exception as exc:
                logger.warning(
                    "Skipping unregistered toolkit %r for agent %s",
                    row["toolkit"],
                    agent_id,
                )
                sentry_sdk.capture_exception(exc)
        return toolkits

    # ------------------------------------------------------------------
    # Agent Skills CRUD
    # ------------------------------------------------------------------

    async def add_skill(
        self, *, account: str, agent_id: str, skill: HarnessSkill
    ) -> str:
        stmt = agent_skills.insert().values(
            account=account,
            agent_id=agent_id,
            skill_id=skill.id,
            name=skill.name,
            description=skill.description,
            instructions=skill.instructions,
            recommended_tools=list(skill.recommended_tools),
            code_examples=list(skill.code_examples),
            preload=skill.preload,
        )
        res = await self.database.execute(stmt)
        return str(res)

    async def patch_skill(
        self,
        *,
        account: str,
        agent_id: str,
        skill_id: str,
        skill: HarnessSkill,
    ) -> None:
        stmt = (
            agent_skills.update()
            .where(
                agent_skills.c.account == account,
                agent_skills.c.agent_id == agent_id,
                agent_skills.c.skill_id == skill_id,
            )
            .values(
                skill_id=skill.id,
                name=skill.name,
                description=skill.description,
                instructions=skill.instructions,
                recommended_tools=list(skill.recommended_tools),
                code_examples=list(skill.code_examples),
                preload=skill.preload,
            )
        )
        await self.database.execute(stmt)

    async def delete_skill(self, *, account: str, agent_id: str, skill_id: str) -> None:
        stmt = agent_skills.delete().where(
            agent_skills.c.account == account,
            agent_skills.c.agent_id == agent_id,
            agent_skills.c.skill_id == skill_id,
        )
        await self.database.execute(stmt)

    async def get_skills(self, account: str, agent_id: str) -> List[HarnessSkill]:
        stmt = (
            agent_skills.select()
            .where(
                agent_skills.c.account == account,
                agent_skills.c.agent_id == agent_id,
            )
            .order_by(agent_skills.c.created.asc())
        )
        rows = await self.database.fetch_all(stmt)
        return [
            HarnessSkill(
                id=row["skill_id"],
                name=row["name"],
                description=row["description"],
                instructions=row["instructions"],
                recommended_tools=tuple(_decode_json_col(row["recommended_tools"], [])),
                code_examples=tuple(_decode_json_col(row["code_examples"], [])),
                preload=bool(row["preload"]),
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Unified Runtime Config Loading
    # ------------------------------------------------------------------

    async def get_agent_config_basic(
        self, account: str, agent_id: str
    ) -> RetrievalAgent:
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
            memory=_decode_json_col(result["memory"], None),
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
    ) -> HarnessAgentConfig:
        agent_row = await self.database.fetch_one(
            retrieval_agent_config.select().where(
                retrieval_agent_config.c.account == account,
                retrieval_agent_config.c.agent_id == agent_id,
            )
        )
        if agent_row is None:
            raise exceptions.NotFoundError("Agent config not found")

        workflow_row = await self.database.fetch_one(
            retrieval_agent_workflow.select().where(
                retrieval_agent_workflow.c.account == account,
                retrieval_agent_workflow.c.agent_id == agent_id,
                retrieval_agent_workflow.c.workflow_id == workflow_id,
                self._active_workflow_condition(),
            )
        )
        if workflow_row is None:
            raise exceptions.NotFoundError("Workflow not found")

        drivers = await self.get_drivers(account, agent_id)
        toolkits = await self.get_toolkits(account, agent_id, workflow_id=workflow_id)
        stored_skills = await self.get_skills(account, agent_id)
        prompts = await self.get_prompts(agent_id, account)

        agent_rules = Rules.model_validate(
            _decode_json_col(agent_row["rules"], {"rules": []})
        )
        wf_rules = Rules.model_validate(
            _decode_json_col(workflow_row["rules"], {"rules": []})
        )
        merged_rules = Rules(rules=[*agent_rules.rules, *wf_rules.rules])

        if default_memory:
            memory = MemoryConfig()
        else:
            raw_mem = _decode_json_col(agent_row["memory"], None)
            memory = (
                MemoryConfig.model_validate(raw_mem)
                if raw_mem is not None
                else MemoryConfig(
                    nucliadb=NucliaDBMemoryConfig(
                        url=internal_nucliadb_url, kbid=agent_id, internal=True
                    )
                )
                if internal_nucliadb_url is not None
                else MemoryConfig()
            )

        harness_settings = dict(_decode_json_col(workflow_row["harness_config"], {}))
        enabled_skill_ids = set(harness_settings.pop("enabled_skills", []) or [])
        skills_by_id: dict[str, HarnessSkill] = {s.id: s for s in stored_skills}
        for sid in enabled_skill_ids:
            reg_entry = GLOBAL_REGISTRY.skills.get(sid)
            if reg_entry and reg_entry.skill is not None and sid not in skills_by_id:
                skills_by_id[sid] = reg_entry.skill

        execution_mode = workflow_row["execution_mode"] or "loop"
        model_id = str(harness_settings.pop("model", "") or "")
        if execution_mode == "loop" and not model_id:
            model_id = llm_defaults.smart

        system_prompt = str(
            harness_settings.pop("system_prompt", None)
            or agent_row["instructions"]
            or SYSTEM_PROMPT
        )

        return HarnessAgentConfig.model_validate(
            {
                "execution_mode": execution_mode,
                "model": model_id,
                "workflow_code": workflow_row["workflow_code"],
                "toolkits": toolkits,
                "drivers": drivers,
                "skills": list(skills_by_id.values()),
                "prompts": prompts,
                "system_prompt": system_prompt,
                "rules": merged_rules,
                "memory": memory,
                **harness_settings,
            }
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
            arguments=(
                [x.model_dump() for x in prompt.arguments]
                if prompt.arguments is not None
                else None
            ),
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
                arguments=(
                    [x.model_dump() for x in prompt.arguments]
                    if prompt.arguments is not None
                    else None
                ),
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

        raw_args = _decode_json_col(result["arguments"], None)
        return PromptConfig(
            name=result["name"],
            description=result["description"],
            prompt=result["prompt"],
            arguments=(
                [PromptArgument.model_validate(arg) for arg in raw_args]
                if raw_args is not None
                else None
            ),
            icons=_decode_json_col(result["icons"], None),
            meta=_decode_json_col(result["meta"], None),
            prompt_id=str(result["id"]),
        )

    async def get_prompts(self, agent_id: str, account: str) -> List[PromptConfig]:
        statement = (
            retrieval_agent_prompts.select()
            .where(retrieval_agent_prompts.c.account == account)
            .where(retrieval_agent_prompts.c.agent_id == agent_id)
        )
        result = await self.database.fetch_all(statement)
        prompts = []
        for row in result:
            raw_args = _decode_json_col(row["arguments"], None)
            prompts.append(
                PromptConfig(
                    name=row["name"],
                    description=row["description"],
                    prompt=row["prompt"],
                    arguments=(
                        [PromptArgument.model_validate(arg) for arg in raw_args]
                        if raw_args is not None
                        else None
                    ),
                    icons=_decode_json_col(row["icons"], None),
                    meta=_decode_json_col(row["meta"], None),
                    prompt_id=str(row["id"]),
                )
            )
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
        if result is None:
            return Rules(rules=[])
        return Rules.model_validate(_decode_json_col(result["rules"], {"rules": []}))

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
        return Rules.model_validate(_decode_json_col(result["rules"], {"rules": []}))

    # ------------------------------------------------------------------
    # Legacy 4-Stage Compatibility Adapter (compiles directly to Harness)
    # ------------------------------------------------------------------

    async def _load_stage_snapshot(
        self, account: str, agent_id: str, workflow_id: str
    ) -> dict[str, list[dict[str, Any]]]:
        row = await self.database.fetch_one(
            retrieval_agent_workflow.select().where(
                retrieval_agent_workflow.c.account == account,
                retrieval_agent_workflow.c.agent_id == agent_id,
                retrieval_agent_workflow.c.workflow_id == workflow_id,
                self._active_workflow_condition(),
            )
        )
        if row is None:
            raise exceptions.NotFoundError("Workflow not found")
        raw = _decode_json_col(row["legacy_stages_snapshot"], None) or {}
        return {
            stage: list(raw.get(stage) or [])
            for stage in ("preprocess", "context", "generation", "postprocess")
        }

    async def _save_and_compile_stages(
        self,
        account: str,
        agent_id: str,
        workflow_id: str,
        snapshot: dict[str, list[dict[str, Any]]],
    ) -> None:
        ext_migrations = {
            k: v.handler
            for k, v in GLOBAL_REGISTRY.migrations.items()
            if v.handler is not None
        }
        compiled = transpile_stages_to_harness(
            preprocess=[
                {"id": x["id"], **(x.get("config") or {})}
                for x in snapshot["preprocess"]
            ],
            context=[
                {"id": x["id"], **(x.get("config") or {})} for x in snapshot["context"]
            ],
            generation=[
                {"id": x["id"], **(x.get("config") or {})}
                for x in snapshot["generation"]
            ],
            postprocess=[
                {"id": x["id"], **(x.get("config") or {})}
                for x in snapshot["postprocess"]
            ],
            external_migrations=ext_migrations,
        )
        async with self.database.transaction():
            await self.database.execute(
                retrieval_agent_workflow.update()
                .where(
                    retrieval_agent_workflow.c.account == account,
                    retrieval_agent_workflow.c.agent_id == agent_id,
                    retrieval_agent_workflow.c.workflow_id == workflow_id,
                )
                .values(
                    execution_mode=compiled["execution_mode"],
                    harness_config=compiled["harness_config"],
                    workflow_code=compiled["workflow_code"],
                    legacy_stages_snapshot=snapshot,
                )
            )
            await self.database.execute(
                agent_workflow_toolkits.delete().where(
                    agent_workflow_toolkits.c.account == account,
                    agent_workflow_toolkits.c.agent_id == agent_id,
                    agent_workflow_toolkits.c.workflow_id == workflow_id,
                )
            )
            for tk in compiled["toolkits"]:
                tk_copy = dict(tk)
                tk_id = str(tk_copy.pop("id", None) or uuid4())
                namespace = str(tk_copy.pop("namespace"))
                provider = str(tk_copy.pop("toolkit"))
                title = tk_copy.pop("title", None)
                lazy_load = bool(tk_copy.pop("lazy_load", False))
                ingest_skills = bool(tk_copy.pop("ingest_skills", True))
                await self.database.execute(
                    agent_workflow_toolkits.insert().values(
                        id=tk_id,
                        account=account,
                        agent_id=agent_id,
                        workflow_id=workflow_id,
                        namespace=namespace,
                        toolkit=provider,
                        title=title,
                        lazy_load=lazy_load,
                        ingest_skills=ingest_skills,
                        config=tk_copy,
                    )
                )

    async def _stage_add(
        self,
        stage: str,
        account: str,
        agent_id: str,
        workflow_id: str,
        agent: BaseModel,
    ) -> str:
        snapshot = await self._load_stage_snapshot(account, agent_id, workflow_id)
        item_id = str(uuid4())
        snapshot[stage].append({"id": item_id, "config": agent.model_dump()})
        await self._save_and_compile_stages(account, agent_id, workflow_id, snapshot)
        return item_id

    async def _stage_patch(
        self,
        stage: str,
        account: str,
        agent_id: str,
        workflow_id: str,
        item_id: str,
        agent: BaseModel,
    ) -> None:
        snapshot = await self._load_stage_snapshot(account, agent_id, workflow_id)
        for entry in snapshot[stage]:
            if str(entry.get("id")) == str(item_id):
                entry["config"] = agent.model_dump()
                break
        await self._save_and_compile_stages(account, agent_id, workflow_id, snapshot)

    async def _stage_delete(
        self, stage: str, account: str, agent_id: str, workflow_id: str, item_id: str
    ) -> None:
        snapshot = await self._load_stage_snapshot(account, agent_id, workflow_id)
        snapshot[stage] = [
            entry for entry in snapshot[stage] if str(entry.get("id")) != str(item_id)
        ]
        await self._save_and_compile_stages(account, agent_id, workflow_id, snapshot)

    async def _stage_get(
        self, stage: str, account: str, agent_id: str, workflow_id: str
    ) -> List[AgentConfig]:
        snapshot = await self._load_stage_snapshot(account, agent_id, workflow_id)
        items: list[AgentConfig] = []
        for entry in snapshot[stage]:
            cfg = get_agent_config_instance(
                agent_config=dict(entry.get("config") or {}),
                agent_type=stage,
            )
            cfg.id = str(entry["id"])
            items.append(cfg)
        return items

    async def add_preprocess(
        self,
        agent_id: str,
        account: str,
        agent: BaseModel,
        workflow_id: str = "default",
    ) -> str:
        return await self._stage_add(
            "preprocess", account, agent_id, workflow_id, agent
        )

    async def patch_preprocess(
        self,
        agent_id: str,
        account: str,
        preprocess: str,
        agent: BaseModel,
        workflow_id: str = "default",
    ) -> None:
        await self._stage_patch(
            "preprocess", account, agent_id, workflow_id, preprocess, agent
        )

    async def delete_preprocess(
        self, account: str, agent_id: str, preprocess: str, workflow_id: str = "default"
    ) -> None:
        await self._stage_delete(
            "preprocess", account, agent_id, workflow_id, preprocess
        )

    async def get_preprocess(
        self, account: str, agent_id: str, workflow_id: str = "default"
    ) -> List[AgentConfig]:
        return await self._stage_get("preprocess", account, agent_id, workflow_id)

    async def add_postprocess(
        self,
        agent_id: str,
        account: str,
        agent: BaseModel,
        workflow_id: str = "default",
    ) -> str:
        return await self._stage_add(
            "postprocess", account, agent_id, workflow_id, agent
        )

    async def patch_postprocess(
        self,
        agent_id: str,
        account: str,
        postprocess: str,
        agent: BaseModel,
        workflow_id: str = "default",
    ) -> None:
        await self._stage_patch(
            "postprocess", account, agent_id, workflow_id, postprocess, agent
        )

    async def delete_postprocess(
        self,
        account: str,
        agent_id: str,
        postprocess: str,
        workflow_id: str = "default",
    ) -> None:
        await self._stage_delete(
            "postprocess", account, agent_id, workflow_id, postprocess
        )

    async def get_postprocess(
        self, account: str, agent_id: str, workflow_id: str = "default"
    ) -> List[AgentConfig]:
        return await self._stage_get("postprocess", account, agent_id, workflow_id)

    async def add_context(
        self,
        agent_id: str,
        account: str,
        agent: BaseModel,
        workflow_id: str = "default",
    ) -> str:
        return await self._stage_add("context", account, agent_id, workflow_id, agent)

    async def patch_context(
        self,
        agent_id: str,
        account: str,
        context: UUID | str,
        agent: BaseModel,
        workflow_id: str = "default",
    ) -> None:
        await self._stage_patch(
            "context", account, agent_id, workflow_id, str(context), agent
        )

    async def delete_context(
        self,
        agent_id: str,
        account: str,
        context: UUID | str,
        workflow_id: str = "default",
    ) -> None:
        await self._stage_delete(
            "context", account, agent_id, workflow_id, str(context)
        )

    async def get_context(
        self, account: str, agent_id: str, workflow_id: str = "default"
    ) -> List[AgentConfig]:
        return await self._stage_get("context", account, agent_id, workflow_id)

    async def add_generation(
        self,
        agent_id: str,
        account: str,
        agent: BaseModel,
        workflow_id: str = "default",
    ) -> str:
        return await self._stage_add(
            "generation", account, agent_id, workflow_id, agent
        )

    async def patch_generation(
        self,
        agent_id: str,
        account: str,
        generation: str,
        agent: BaseModel,
        workflow_id: str = "default",
    ) -> None:
        await self._stage_patch(
            "generation", account, agent_id, workflow_id, generation, agent
        )

    async def delete_generation(
        self, agent_id: str, account: str, generation: str, workflow_id: str = "default"
    ) -> None:
        await self._stage_delete(
            "generation", account, agent_id, workflow_id, generation
        )

    async def get_generation(
        self, account: str, agent_id: str, workflow_id: str = "default"
    ) -> List[AgentConfig]:
        return await self._stage_get("generation", account, agent_id, workflow_id)

    # ------------------------------------------------------------------
    # Export / Import (V2 native + V1 legacy auto-migration)
    # ------------------------------------------------------------------

    async def export(
        self, account: str, agent_id: str, passphrase: str
    ) -> tuple[bytes, str | None]:
        if len(passphrase) < 16:
            raise Exception("Passphrase too short, minimum 16 characters required")
        workflow_config: dict[str, HarnessAgentConfig] = {}
        default_config = None
        workflows = [wf.id for wf in await self.workflows_list(account, agent_id)]
        for workflow in workflows:
            try:
                cfg = await self.get_agent_config(
                    account, agent_id, default_memory=True, workflow_id=workflow
                )
                if workflow == "default":
                    default_config = cfg
                workflow_config[workflow] = cfg
            except Exception as exc:
                logger.exception("Agent not found for export")
                raise Exception("Retrieval agent not found for export") from exc

        export_model = HarnessAgentExportV2(
            agent_config=default_config,
            agent_config_workflows=workflow_config,
            skills=await self.get_skills(account, agent_id),
            prompts=await self.get_prompts(agent_id, account),
        )
        export_bytes = export_model.model_dump_json().encode("utf-8")
        key, salt = fernet_key_from_passphrase(passphrase, None)
        fernet = Fernet(key)
        try:
            encrypted_bytes = salt + fernet.encrypt(export_bytes)
        except Exception as exc:
            logger.exception("Error encrypting agent export")
            raise Exception("Error encrypting export") from exc

        return encrypted_bytes, None

    async def import_config(
        self,
        account: str,
        agent_id: str,
        import_file: UploadFile,
        passphrase: str,
        overwrite: bool,
    ):
        destination_agent_config = await self.get_agent_config(
            account, agent_id, default_memory=True, workflow_id="default"
        )
        if not destination_agent_config.is_empty():
            if not overwrite:
                raise exceptions.InvalidTargetAgentError()
            await self.delete_agent(account=account, agent_id=agent_id)
            await self.add_agent(
                account=account,
                agent_id=agent_id,
                memory=MemoryConfig(),
                rules=Rules(rules=[]),
            )

        salt = await import_file.read(16)
        key, _ = fernet_key_from_passphrase(passphrase, salt)
        fernet = Fernet(key)
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

        try:
            decrypted_bytes = fernet.decrypt(bytes(import_bytes))
        except Exception as e:
            raise exceptions.ExportEncryptionError from e

        try:
            parsed_export = _export_union_adapter().validate_json(
                decrypted_bytes.decode("utf-8")
            )
        except ValidationError as e:
            raise exceptions.ParseExportError from e

        if isinstance(parsed_export, HarnessAgentExportV2):
            await self._import_v2(account, agent_id, parsed_export)
        elif isinstance(parsed_export, RetrievalAgentExportV1):
            await self._import_v1_as_harness(account, agent_id, parsed_export)
        else:
            raise exceptions.ParseExportError("Unsupported export version")

    async def _import_v2(
        self, account: str, agent_id: str, parsed: HarnessAgentExportV2
    ) -> None:
        workflow_configs = dict(parsed.agent_config_workflows)
        if parsed.agent_config is not None:
            workflow_configs.setdefault("default", parsed.agent_config)
        drivers: dict[str, DriverConfig] = {}
        agent_rules: Rules | None = None
        try:
            for wf_id, wf_cfg in workflow_configs.items():
                for drv in wf_cfg.drivers:
                    drivers[drv.identifier] = drv
                agent_rules = wf_cfg.rules
                harness_dict = {
                    "model": wf_cfg.model,
                    "system_prompt": wf_cfg.system_prompt,
                    "feedback_enabled": wf_cfg.feedback_enabled,
                    "codemode_enabled": wf_cfg.codemode_enabled,
                }
                if wf_id == "default":
                    await self.set_workflow_harness(
                        account,
                        agent_id,
                        wf_id,
                        WorkflowHarnessUpdate(
                            execution_mode=wf_cfg.execution_mode,
                            harness_config=harness_dict,
                            workflow_code=wf_cfg.workflow_code,
                        ),
                    )
                else:
                    await self.add_workflow(
                        account,
                        agent_id,
                        WorkflowInput(
                            id=wf_id,
                            name=wf_id,
                            execution_mode=wf_cfg.execution_mode,
                            harness_config=harness_dict,
                            workflow_code=wf_cfg.workflow_code,
                        ),
                    )
                for tk in wf_cfg.toolkits:
                    await self.add_toolkit(
                        account=account,
                        agent_id=agent_id,
                        toolkit=tk,
                        workflow_id=wf_id,
                    )
            if agent_rules is not None:
                await self.set_rules(
                    agent_id=agent_id, account=account, rules=agent_rules
                )
            for drv in drivers.values():
                await self.add_driver(agent_id=agent_id, account=account, config=drv)
            for sk in parsed.skills:
                await self.add_skill(account=account, agent_id=agent_id, skill=sk)
            for pr in parsed.prompts:
                await self.add_prompt(agent_id=agent_id, account=account, prompt=pr)
        except Exception as e:
            raise exceptions.ParseExportError(
                "Failed to import Harness agent configuration"
            ) from e

    async def _import_v1_as_harness(
        self, account: str, agent_id: str, parsed: RetrievalAgentExportV1
    ) -> None:
        workflow_configs: dict[str, RetrievalAgentConfig] = {}
        if parsed.agent_config is not None:
            workflow_configs["default"] = parsed.agent_config
        workflow_configs.update(parsed.agent_config_workflows)
        drivers: dict[str, DriverConfig] = {}
        agent_rules: Rules | None = None
        try:
            for wf_id, legacy_wf in workflow_configs.items():
                for drv in legacy_wf.drivers:
                    drivers[drv.identifier] = drv
                agent_rules = legacy_wf.rules
                wf = legacy_wf.workflow
                if wf_id == "default":
                    await self.set_workflow(
                        account=account,
                        agent_id=agent_id,
                        workflow_id=wf_id,
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
                            id=wf_id,
                            name=wf.name,
                            description=wf.description,
                            parameters=wf.parameters,
                            rules=wf.rules or Rules(rules=[]),
                        ),
                    )
                snapshot = {
                    stage: [
                        {"id": str(uuid4()), "config": item.model_dump()}
                        for item in getattr(legacy_wf, stage)
                    ]
                    for stage in ("preprocess", "context", "generation", "postprocess")
                }
                await self._save_and_compile_stages(account, agent_id, wf_id, snapshot)

            if agent_rules is not None:
                await self.set_rules(
                    agent_id=agent_id, account=account, rules=agent_rules
                )
            for drv in drivers.values():
                await self.add_driver(agent_id=agent_id, account=account, config=drv)
            for pr in parsed.prompts:
                await self.add_prompt(agent_id=agent_id, account=account, prompt=pr)
        except Exception as e:
            raise exceptions.ParseExportError(
                "Failed to import retrieval agent configuration"
            ) from e


def update_driver_config(
    config: DriverConfig, previous_config: DriverConfig
) -> DriverConfig:
    desired_configuration = config.model_dump(exclude_unset=True)
    updated_configuration = previous_config.model_dump()
    return config.__class__(**deep_update(updated_configuration, desired_configuration))


def deep_update(original: dict, updates: dict) -> dict:
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
