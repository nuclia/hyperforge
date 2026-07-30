"""gRPC serving interface for the A2A protocol.

Hosts the A2A ``A2AService`` gRPC servicer (via ``GrpcHandler``) wrapping a
``DefaultRequestHandler`` driven by :class:`HyperforgeA2AExecutor`. The server
shares the broker + agent manager with the rest of Hyperforge so every A2A
interaction flows through the same worker pipeline as the HTTP/WS API.
"""

from concurrent import futures
from typing import Any

import grpc
from a2a.server.request_handlers import DefaultRequestHandler, GrpcHandler
from a2a.types import a2a_pb2_grpc

from hyperforge.a2a import logger
from hyperforge.a2a.card import build_agent_card, build_agent_skills
from hyperforge.a2a.context import A2AServerContext
from hyperforge.a2a.executor import HyperforgeA2AExecutor
from hyperforge.a2a.settings import A2ASettings
from hyperforge.a2a.task_store import RedisA2ASDKTaskStore, RedisA2ATaskStore
from hyperforge.broker.redis import RedisBroker
from hyperforge.configure import GLOBAL_REGISTRY, load_all_configurations, scan
from hyperforge.db.agents import AgentManager
from hyperforge.db.settings import DataManagerSettings


def _load_modules(settings: A2ASettings) -> None:
    for load_module in settings.load_modules:
        try:
            scan(load_module)
            load_all_configurations(load_module)
        except ImportError:
            logger.error(f"Module {load_module} could not be loaded")


def build_server_credentials(settings: A2ASettings) -> grpc.ServerCredentials:
    """Load the configured server certificate and optional mTLS CA at startup."""
    if (
        not settings.a2a_tls_certificate_chain_path
        or not settings.a2a_tls_private_key_path
    ):
        raise ValueError("A2A TLS certificate chain and private key must be configured")
    certificate_chain = settings.a2a_tls_certificate_chain_path.read_bytes()
    private_key = settings.a2a_tls_private_key_path.read_bytes()
    client_ca = (
        settings.a2a_tls_client_ca_path.read_bytes()
        if settings.a2a_tls_client_ca_path
        else None
    )
    return grpc.ssl_server_credentials(
        [(private_key, certificate_chain)],
        root_certificates=client_ca,
        require_client_auth=client_ca is not None,
    )


async def build_grpc_server(
    settings: A2ASettings,
    data_manager_settings: DataManagerSettings,
) -> tuple[grpc.aio.Server, AgentManager, RedisBroker]:
    """Wire up SaaS dependencies and the A2A gRPC servicer."""
    if not settings.a2a_account or not settings.a2a_agent_id:
        raise ValueError("A2A_ACCOUNT and A2A_AGENT_ID must be configured")

    GLOBAL_REGISTRY.clear()

    broker = RedisBroker.from_url(
        url=settings.valkey_url,
        activate_subject=settings.activate_subject,
        keepalive_ms=int(settings.pubsub_keepalive_seconds * 1000),
        cluster_mode=settings.valkey_cluster_mode,
    )

    agent_manager = await AgentManager.from_settings(settings=data_manager_settings)
    await agent_manager.initialize()

    _load_modules(settings)

    try:
        server = await build_grpc_server_from_runtime(settings, agent_manager, broker)
    except Exception:
        await agent_manager.finalize()
        await broker.finalize()
        raise

    return server, agent_manager, broker


async def build_grpc_server_from_runtime(
    settings: A2ASettings,
    agent_manager: Any,
    broker: RedisBroker,
) -> grpc.aio.Server:
    """Build an A2A server from already-initialized Hyperforge runtime services."""
    if not settings.a2a_account or not settings.a2a_agent_id:
        raise ValueError("A2A_ACCOUNT and A2A_AGENT_ID must be configured")

    app_context = A2AServerContext(
        settings=settings,
        agent_manager=agent_manager,
        broker=broker,
        task_store=RedisA2ATaskStore(
            broker.client,
            settings.a2a_task_store_prefix,
            settings.a2a_task_ttl_seconds,
        ),
    )

    executor = HyperforgeA2AExecutor(app_context)
    skills = await build_agent_skills(
        agent_manager, settings.a2a_account, settings.a2a_agent_id
    )
    if not skills:
        raise ValueError(
            "A2A server agent must have at least one workflow to advertise"
        )
    agent_card = build_agent_card(settings, skills)
    task_owner = f"{settings.a2a_account}:{settings.a2a_agent_id}"
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=RedisA2ASDKTaskStore(
            broker.client,
            settings.a2a_task_store_prefix,
            settings.a2a_task_ttl_seconds,
            owner_resolver=lambda _context: task_owner,
        ),
        agent_card=agent_card,
    )
    grpc_handler = GrpcHandler(request_handler=request_handler)

    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=settings.a2a_grpc_max_workers)
    )
    a2a_pb2_grpc.add_A2AServiceServicer_to_server(grpc_handler, server)
    bind_address = f"{settings.a2a_grpc_host}:{settings.a2a_grpc_port}"
    if settings.a2a_tls_enabled:
        bound_port = server.add_secure_port(
            bind_address,
            build_server_credentials(settings),
        )
    else:
        bound_port = server.add_insecure_port(bind_address)
    if bound_port == 0:
        raise RuntimeError(f"Unable to bind A2A gRPC server to {bind_address}")

    return server


async def serve(
    settings: A2ASettings,
    data_manager_settings: DataManagerSettings,
) -> None:
    server, agent_manager, broker = await build_grpc_server(
        settings, data_manager_settings
    )
    await server.start()
    logger.warning(
        f"A2A gRPC server listening on "
        f"{settings.a2a_grpc_host}:{settings.a2a_grpc_port} "
        f"(tls={settings.a2a_tls_enabled})"
    )
    try:
        await server.wait_for_termination()
    finally:
        await server.stop(grace=5)
        await agent_manager.finalize()
        await broker.finalize()
        GLOBAL_REGISTRY.clear()
