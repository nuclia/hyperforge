import uvicorn
from hyperforge.db.settings import DataManagerSettings
from nucliadb_telemetry.fastapi import instrument_app
from nucliadb_telemetry.logs import setup_logging
from nucliadb_telemetry.utils import get_telemetry

from hyperforge import openapi
from nucliadb_agentic_api import SERVICE_NAME
from nucliadb_agentic_api.app import HTTPApplication
from nucliadb_agentic_api.settings import Settings
from nucliadb_agentic_api.v1.router import router


def run():  # pragma: no cover
    setup_logging()
    settings = Settings()
    data_manager_settings = DataManagerSettings()
    app = HTTPApplication(
        settings,
        data_manager_settings=data_manager_settings,
    )
    instrument_app(
        app,
        tracer_provider=get_telemetry(SERVICE_NAME),
        excluded_urls=["/", "/metrics", "/health/ready", "/health/alive"],
        metrics=True,
        trace_id_on_responses=True,
    )
    uvicorn.run(app, host=settings.http_host, port=settings.http_port)


def extract_openapi():
    openapi.extract_openapi_command(
        "nucliadb_agentic_api", "NucliaDB Agentic API", router
    )
