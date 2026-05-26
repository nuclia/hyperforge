import sys

import uvicorn
import yaml
from hyperforge_standalone import logger
from hyperforge_standalone.app import StandaloneApplication
from hyperforge_standalone.config import StandAloneAgentConfig, StandaloneConfig
from hyperforge_standalone.settings import StandaloneSettings

from hyperforge.configure import resolve_dotted_name


def run(
    application_class: type[StandaloneApplication] | None = None,
) -> None:  # pragma: no cover
    settings: StandaloneSettings = StandaloneSettings()  # type: ignore[call-arg]

    if not settings.agents_config.exists():
        print(
            f"error: agents config file not found: {settings.agents_config}",
            file=sys.stderr,
        )
        sys.exit(1)

    from hyperforge.configure import load_all_configurations, scan

    # Register all built-in agents and drivers (same as the base initialize,
    # but without start_health_check() — the FastAPI app handles /health/*).
    scan("nuclia_agents.agents.agents")
    scan("nuclia_agents.drivers.drivers")
    load_all_configurations("nuclia_agents")

    for load_module in settings.load_modules:
        try:
            scan(load_module)
            load_all_configurations(load_module)
        except ImportError:
            logger.error(f"Module {load_module} could not be loaded")

    if settings.agents_config.suffix == ".yaml":
        agents_cfg: dict[str, StandAloneAgentConfig] = StandaloneConfig.validate_python(
            yaml.safe_load(settings.agents_config.read_text())
        )
    else:
        agents_cfg = StandaloneConfig.validate_json(settings.agents_config.read_text())

    if application_class is None:
        application_class = resolve_dotted_name(settings.standalone_application_class)
    app = application_class(agents_cfg, settings)

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
