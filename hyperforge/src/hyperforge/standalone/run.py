import sys

import uvicorn
import yaml

from hyperforge.configure import resolve_dotted_name
from hyperforge.standalone import logger
from hyperforge.standalone.app import StandaloneApplication
from hyperforge.standalone.config import StandAloneAgentConfig, StandaloneConfig
from hyperforge.standalone.settings import StandaloneSettings

BUILTIN_MODULES = (
    "hyperforge_a2a",
    "hyperforge_conditional",
    "hyperforge_external",
    "hyperforge_generate.agent",
    "hyperforge_google",
    "hyperforge_historical",
    "hyperforge_http",
    "hyperforge_mcp",
    "hyperforge_mcp.stdio",
    "hyperforge_nucliadb",
    "hyperforge_passthrough",
    "hyperforge_perplexity",
    "hyperforge_perplexity_search",
    "hyperforge_related.agent",
    "hyperforge_remi",
    "hyperforge_rephrase",
    "hyperforge_restart.agent",
    "hyperforge_restricted",
    "hyperforge_smart",
    "hyperforge_static",
    "hyperforge_static_string",
    "hyperforge_summarize",
)


def load_hyperforge_builtins() -> None:
    """Load every installed built-in Hyperforge agent and driver package."""
    from hyperforge.configure import load_all_configurations, scan

    loaded_packages: set[str] = set()
    for module in BUILTIN_MODULES:
        package = module.partition(".")[0]
        try:
            scan(module)
        except ModuleNotFoundError as exc:
            # Standalone consumers may install only the built-ins they use. Do not
            # hide missing dependencies from a built-in package that is installed.
            if exc.name == package:
                continue
            raise
        loaded_packages.add(package)

    for package in loaded_packages:
        load_all_configurations(package)


def load_default_modules() -> str:
    """
    Load built-in agent/driver modules.

    Prefer nuclia_agents when available. In generic standalone deployments where
    nuclia_agents is not installed, fall back to Hyperforge built-ins.
    """
    from hyperforge.configure import load_all_configurations, scan

    try:
        scan("nuclia_agents.agents.agents")
        scan("nuclia_agents.drivers.drivers")
        load_all_configurations("nuclia_agents")
        return "nuclia_agents"
    except ModuleNotFoundError as exc:
        if not exc.name or exc.name.startswith("nuclia_agents"):
            logger.info(
                "nuclia_agents package not available; loading Hyperforge built-ins"
            )
            load_hyperforge_builtins()
            return "hyperforge"
        raise


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
    load_default_modules()

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
