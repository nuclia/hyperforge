"""Entry point for the Hyperforge A2A gRPC server (``hyperforge-a2a-grpc``)."""

import asyncio

from nucliadb_telemetry.logs import setup_logging
from nucliadb_telemetry.settings import LogLevel, LogSettings

from hyperforge.a2a.server import serve
from hyperforge.a2a.settings import A2ASettings
from hyperforge.db.settings import DataManagerSettings


def run() -> None:  # pragma: no cover
    settings = A2ASettings()
    setup_logging(
        settings=LogSettings(
            debug=settings.debug,
            log_level=LogLevel(settings.log_level),
            logger_levels={
                "hyperforge.configure": LogLevel.WARNING,
            },
        )
    )
    data_manager_settings = DataManagerSettings()
    asyncio.run(serve(settings, data_manager_settings))


if __name__ == "__main__":  # pragma: no cover
    run()
