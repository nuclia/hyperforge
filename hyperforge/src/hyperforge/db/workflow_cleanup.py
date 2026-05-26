import asyncio
import datetime
from importlib.metadata import version

from nucliadb_telemetry import errors
from nucliadb_telemetry.errors import setup_error_handling
from nucliadb_telemetry.logs import setup_logging
from nucliadb_telemetry.settings import LogLevel, LogSettings

from hyperforge.db import logger
from hyperforge.db.agents import WORKFLOW_PURGE_RETENTION, AgentManager
from hyperforge.db.settings import DataManagerSettings


async def cleanup_deleted_workflows(
    manager: AgentManager,
    older_than: datetime.timedelta = WORKFLOW_PURGE_RETENTION,
):
    logger.info("Cleaning up deleted workflows")
    workflows = await manager.get_expired_deleted_workflows(older_than=older_than)
    logger.info("Found deleted workflows to clean up", extra={"count": len(workflows)})

    for workflow in workflows:
        try:
            logger.info(
                "Purging deleted workflow",
                extra={
                    "account": workflow["account"],
                    "agent_id": workflow["agent_id"],
                    "workflow_id": workflow["workflow_id"],
                },
            )
            await manager.purge_deleted_workflow(
                account=workflow["account"],
                agent_id=workflow["agent_id"],
                workflow_id=workflow["workflow_id"],
            )
        except Exception as exc:
            errors.capture_exception(exc)
            logger.error(
                "Failed to purge deleted workflow",
                exc_info=exc,
                extra={
                    "account": workflow["account"],
                    "agent_id": workflow["agent_id"],
                    "workflow_id": workflow["workflow_id"],
                },
            )


async def cronjob(manager: AgentManager):
    await cleanup_deleted_workflows(manager)


def run():  # pragma: no cover
    asyncio.run(_main())


async def _main():
    log_settings = LogSettings(logger_levels={"hyperforge.server": LogLevel.INFO})
    setup_logging(settings=log_settings)
    setup_error_handling(version("hyperforge"))
    data_manager_settings = DataManagerSettings()
    manager = await AgentManager.from_settings(settings=data_manager_settings)
    await manager.initialize()
    try:
        await cronjob(manager)
    finally:
        await manager.finalize()
