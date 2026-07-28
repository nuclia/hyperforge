"""Agent card construction for the Hyperforge A2A gRPC interface.

The A2A agent card advertises the service, its transport interface and the
skills it exposes. Hyperforge exposes each configured *workflow* of an agent as
an A2A :class:`AgentSkill`, mirroring how the MCP interface exposes workflows as
tools (see ``hyperforge.api.v1.mcp_interaction.list_tools``).
"""

from typing import Optional

from a2a.types import a2a_pb2
from a2a.utils import TransportProtocol

from hyperforge.a2a.settings import A2ASettings
from hyperforge.db.agents import AgentManager
from hyperforge.workflows import WorkflowData


def _public_url(settings: A2ASettings) -> str:
    if settings.a2a_public_url:
        return settings.a2a_public_url
    return f"{settings.a2a_grpc_host}:{settings.a2a_grpc_port}"


def skill_from_workflow(agent_id: str, workflow: WorkflowData) -> a2a_pb2.AgentSkill:
    """Map a single Hyperforge workflow onto an A2A skill."""
    return a2a_pb2.AgentSkill(
        id=f"{agent_id}:{workflow.id}",
        name=workflow.name,
        description=workflow.description or workflow.name,
        tags=["hyperforge", "workflow", agent_id],
        input_modes=["text/plain"],
        output_modes=["text/plain"],
    )


async def build_agent_skills(
    agent_manager: AgentManager, account: str, agent_id: str
) -> list[a2a_pb2.AgentSkill]:
    """Enumerate the workflows of an agent as A2A skills."""
    workflows = await agent_manager.workflows_list(account=account, agent_id=agent_id)
    return [skill_from_workflow(agent_id, workflow) for workflow in workflows]


def build_agent_card(
    settings: A2ASettings,
    skills: Optional[list[a2a_pb2.AgentSkill]] = None,
) -> a2a_pb2.AgentCard:
    """Build the A2A agent card advertised by the gRPC server.

    Callers without a configured agent may omit ``skills`` to create a generic
    card. The production server always supplies workflow-derived skills.
    """
    url = _public_url(settings)

    if skills is None:
        skills = [
            a2a_pb2.AgentSkill(
                id="ask",
                name="ask",
                description=(
                    "Ask a question to a Hyperforge agent. Provide the target "
                    "'account', 'agent_id' and optional 'workflow_id' in the "
                    "message metadata."
                ),
                tags=["hyperforge", "rag"],
                input_modes=["text/plain"],
                output_modes=["text/plain"],
            )
        ]

    return a2a_pb2.AgentCard(
        name=settings.a2a_agent_name,
        description=settings.a2a_agent_description,
        version=settings.a2a_agent_version,
        provider=a2a_pb2.AgentProvider(organization="Nuclia", url="https://nuclia.com"),
        capabilities=a2a_pb2.AgentCapabilities(
            streaming=True,
            push_notifications=False,
        ),
        supported_interfaces=[
            a2a_pb2.AgentInterface(
                url=url,
                protocol_binding=TransportProtocol.GRPC,
            )
        ],
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=skills,
    )
