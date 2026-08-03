"""Agent card construction for the Hyperforge A2A gRPC interface.

The A2A agent card advertises the service, its transport interface and the
skills it exposes. Hyperforge exposes each configured *workflow* of an agent as
an A2A :class:`AgentSkill`, mirroring how the MCP interface exposes workflows as
tools (see ``hyperforge.api.v1.mcp_interaction.list_tools``).
"""

from typing import Any, Optional

from a2a.types import a2a_pb2
from a2a.utils import TransportProtocol

from hyperforge.a2a.settings import A2ASettings
from hyperforge.workflows import WorkflowData


def _public_url(settings: A2ASettings) -> str:
    if settings.a2a_public_url:
        return settings.a2a_public_url
    if settings.a2a_tls_enabled:
        raise ValueError("A2A_PUBLIC_URL is required when A2A TLS is enabled")
    if settings.a2a_grpc_host in {"0.0.0.0", "::"}:
        raise ValueError(
            "A2A_PUBLIC_URL is required when A2A_GRPC_HOST is a wildcard address"
        )
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
    agent_manager: Any, account: str, agent_id: str
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

    card = a2a_pb2.AgentCard(
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
    if settings.a2a_auth_enabled:
        card.security_schemes["bearer"].CopyFrom(
            a2a_pb2.SecurityScheme(
                http_auth_security_scheme=a2a_pb2.HTTPAuthSecurityScheme(
                    scheme="bearer"
                )
            )
        )
        card.security_requirements.append(
            a2a_pb2.SecurityRequirement(schemes={"bearer": a2a_pb2.StringList()})
        )
    return card
