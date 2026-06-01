from typing import Any, Dict, Tuple
from uuid import uuid4

from hyperforge.configure import get_driver_klass
from hyperforge.driver import Driver
from hyperforge.models import MemoryConfig, Rules
from hyperforge.retrieval.config import RetrievalAgentConfig
from hyperforge.workflows import WorkflowData
from hyperforge_mcp.config import MCPAgentConfig, Transport
from hyperforge_mcp.config_driver import MCPHTTPDriverConfig, MCPHTTPInnerConfig
from hyperforge_nucliadb.basic_ask_config import BasicAskAgentConfig
from hyperforge_nucliadb.driver_config import NucliaDBConfig, NucliaDBConnection
from hyperforge_rephrase.config import RephraseAgentConfig
from hyperforge_smart.config import SmartAgentConfig
from hyperforge_summarize.config import SummarizeAgentConfig
from nucliadb_models.search import AskRequest

from nucliadb_agentic_api.models import AgenticConfigSchema


async def transform_agentic_config(
    agentic_config: AgenticConfigSchema,
    global_drivers: Dict[str, Driver],
    ask_request: AskRequest,
    resource: str | None = None,
) -> Tuple[RetrievalAgentConfig, Dict[str, Driver]]:
    drivers = {}

    title = agentic_config.title if agentic_config.title else "Default Agentic Config"

    if agentic_config.config.rephrase:
        config: Dict[str, Any] = {
            "title": f"{title} - Retrieval",
        }
        if agentic_config.config.rephrase.model:
            config["model"] = agentic_config.config.rephrase.model
        if agentic_config.config.rephrase.prompt:
            config["rules"] = [agentic_config.config.rephrase.prompt]
        if agentic_config.config.rephrase.ask_to:
            # Same KB with different layers
            config["kbid"] = agentic_config.config.rephrase.ask_to
        preprocess = [RephraseAgentConfig(**config)]
    else:
        preprocess = []

    if agentic_config.config.smart_agent:
        config = {
            "title": f"{title} - Smart Agent",
        }
        if agentic_config.config.smart_agent.extra_prompt:
            config["extra_prompt"] = agentic_config.config.smart_agent.extra_prompt
        if agentic_config.config.smart_agent.models:
            if agentic_config.config.smart_agent.models.context_validation:
                config["context_validation_model"] = (
                    agentic_config.config.smart_agent.models.context_validation
                )
            if agentic_config.config.smart_agent.models.planner:
                config["planner_model"] = (
                    agentic_config.config.smart_agent.models.planner
                )
            if agentic_config.config.smart_agent.models.executor:
                config["executor_model"] = (
                    agentic_config.config.smart_agent.models.executor
                )
        registered_agents = []
        for source in agentic_config.config.smart_agent.sources:
            source_config: Dict[str, Any] = {
                "title": f"{title} - Smart Agent - {source.type.capitalize()} Source",
            }

            if source.type == "nucliadb":
                # Same KB different
                uid = uuid4().hex
                source_config["sources"] = [uid]
                ask_ndb_agent_config = BasicAskAgentConfig(**source_config)
                ndb_driver_config = NucliaDBConfig(
                    identifier=uid,
                    name="NucliaDB",
                    provider="nucliadb",
                    config=NucliaDBConnection(
                        url="",  # TODO: pass real URL if needed
                        manager="",
                        description="",
                        kbid=kbid,
                    ),
                )  # TODO: pass real config if needed
                driver_class = get_driver_klass(
                    "nucliadb"
                )  # Check if driver provider is valid
                drivers[uid] = await driver_class.init(ndb_driver_config)
                registered_agents.append(ask_ndb_agent_config)
            elif source.type == "mcp":
                uid = uuid4().hex
                source_config["sources"] = [uid]
                source_config["transport"] = Transport.HTTP
                ask_ndb_agent_config = MCPAgentConfig(**source_config)
                ndb_driver_config = MCPHTTPDriverConfig(
                    identifier=uid,
                    name="MCPHTTP",
                    provider="mcphttp",
                    config=MCPHTTPInnerConfig(
                        uri=source.uri,  # TODO: pass real URL if needed
                        headers=source.headers if source.headers else {},
                    ),
                )  # TODO: pass real config if needed
                driver_class = get_driver_klass(
                    "mcphttp"
                )  # Check if driver provider is valid
                drivers[uid] = await driver_class.init(ndb_driver_config)
                registered_agents.append(ask_ndb_agent_config)

            elif source.type == "google":
                google_agent_config = GoogleAgentConfig(**source_config)

                drivers[uid] = global_drivers["google"]  # type: ignore
                registered_agents.append(google_agent_config)

            elif source.type == "perplexity":
                perplexity_agent_config = PerplexityAgentConfig(**source_config)
                drivers[uid] = global_drivers["perplexity"]  # type: ignore
                registered_agents.append(perplexity_agent_config)

        config["registered_agents"] = registered_agents
        context = [SmartAgentConfig(**config)]
    else:
        context = []

    if agentic_config.config.summarize:
        config = {
            "title": f"{title} - Summarize Agent",
        }
        if agentic_config.config.summarize.model:
            config["model"] = agentic_config.config.summarize.model
        if agentic_config.config.summarize.user_prompt:
            config["user_prompt"] = agentic_config.config.summarize.user_prompt
        if agentic_config.config.summarize.system_prompt:
            config["system_prompt"] = agentic_config.config.summarize.system_prompt
        if agentic_config.config.summarize.conversational:
            config["conversational"] = agentic_config.config.summarize.conversational
        if agentic_config.config.summarize.model:
            config["model"] = agentic_config.config.summarize.model

        generation = [SummarizeAgentConfig(**config)]
    else:
        generation = []

    return RetrievalAgentConfig(
        drivers=[],
        rules=Rules(),
        memory=MemoryConfig(),
        workflow=WorkflowData(
            id="default",
            name="Default Workflow",
            description="Default workflow for agentic config transformation",
            parameters=None,
        ),
        preprocess=preprocess,  # type: ignore
        context=context,  # type: ignore
        generation=generation,  # type: ignore
        postprocess=[],
    ), drivers
