import uvicorn
from nucliadb_telemetry.fastapi import instrument_app
from nucliadb_telemetry.logs import setup_logging
from nucliadb_telemetry.utils import get_telemetry

from hyperforge import openapi
from hyperforge.api import SERVICE_NAME
from hyperforge.api.app import HTTPApplication
from hyperforge.api.settings import Settings
from hyperforge.api.v1.router import router
from hyperforge.db.settings import DataManagerSettings


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
    openapi.extract_openapi_command("arag", "ARAG API", router)
