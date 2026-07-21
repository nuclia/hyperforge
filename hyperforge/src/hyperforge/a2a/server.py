"""gRPC serving interface for the A2A protocol.

Hosts the A2A ``A2AService`` gRPC servicer (via ``GrpcHandler``) wrapping a
``DefaultRequestHandler`` driven by :class:`HyperforgeA2AExecutor`. The server
shares the broker + agent manager with the rest of Hyperforge so every A2A
interaction flows through the same worker pipeline as the HTTP/WS API.
"""

from concurrent import futures

import grpc
from a2a.server.request_handlers import DefaultRequestHandler, GrpcHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import a2a_pb2_grpc

from hyperforge.a2a import logger
from hyperforge.a2a.card import build_agent_card
from hyperforge.a2a.context import A2AServerContext
from hyperforge.a2a.executor import HyperforgeA2AExecutor
from hyperforge.a2a.settings import A2ASettings
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


async def build_grpc_server(
    settings: A2ASettings,
    data_manager_settings: DataManagerSettings,
) -> tuple[grpc.aio.Server, AgentManager, RedisBroker]:
    """Wire up the broker, agent manager and A2A gRPC servicer."""
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

    app_context = A2AServerContext(
        settings=settings,
        agent_manager=agent_manager,
        broker=broker,
    )

    executor = HyperforgeA2AExecutor(app_context)
    agent_card = build_agent_card(settings)
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )
    grpc_handler = GrpcHandler(request_handler=request_handler)

    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=settings.a2a_grpc_max_workers)
    )
    a2a_pb2_grpc.add_A2AServiceServicer_to_server(grpc_handler, server)
    server.add_insecure_port(f"{settings.a2a_grpc_host}:{settings.a2a_grpc_port}")

    return server, agent_manager, broker


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
        f"{settings.a2a_grpc_host}:{settings.a2a_grpc_port}"
    )
    try:
        await server.wait_for_termination()
    finally:
        await server.stop(grace=5)
        await agent_manager.finalize()
        await broker.finalize()
        GLOBAL_REGISTRY.clear()
