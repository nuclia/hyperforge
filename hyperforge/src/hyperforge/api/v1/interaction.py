import asyncio
import uuid
from enum import Enum
from typing import TYPE_CHECKING, AsyncIterator, Optional

import opentelemetry.propagate
from fastapi import (
    Header,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import StreamingResponse
from nucliadb_sdk import NucliaDBAsync
from pydantic import ValidationError

from hyperforge import logger
from hyperforge.api.authentication import requires_one
from hyperforge.api.models import AgentRole, InteractionRequest
from hyperforge.api.session import create_session_resource, resolve_session_id
from hyperforge.api.settings import Settings
from hyperforge.api.utils import agent_has_nucliadb_memory
from hyperforge.api.v1.router import router
from hyperforge.api.v1.utils import tracer
from hyperforge.broker import AgentTimeoutError
from hyperforge.db import exceptions
from hyperforge.db.agents import AgentManager
from hyperforge.interaction import (
    AnswerOperation,
    AragAnswer,
    ARAGException,
)
from hyperforge.pubsub import (
    AgentAnswer,
    AgentDone,
    AgentPing,
    AgentToUserRequest,
    OAuthRequest,
    StartInteraction,
    UserToAgentInteraction,
)

if TYPE_CHECKING:
    from hyperforge.api.app import HTTPApplication


async def ensure_session_exists(
    ndb: NucliaDBAsync,
    agent_id: str,
    session: str,
    create_if_not_exists: bool,
) -> tuple[str, Optional[str]]:
    """
    Check if session exists and create it if needed.

    Returns:
        The effective session id and None if the session exists or was created.
        The original session id and an error message if the session doesn't exist
        and shouldn't be created.
    """
    # Check if session exists. The client may pass either the resource UUID or
    # the session slug, but the memory layer needs the NucliaDB resource UUID.
    existing_session_id = await resolve_session_id(ndb, agent_id, session)
    if existing_session_id is not None:
        return existing_session_id, None

    # Session doesn't exist
    if not create_if_not_exists:
        return session, f"Session '{session}' does not exist"

    # Create the session
    try:
        created = await create_session_resource(
            ndb=ndb,
            agent_id=agent_id,
            slug=session,
            title=f"Session {session}",
            summary="Auto-created session",
            data="",
        )
        return created.uuid, None
    except Exception as e:
        logger.exception(f"Error creating session {session} for agent {agent_id}: {e}")
        return session, f"Failed to create session: {str(e)}"


class Shutdown:
    pass


class WebsocketReceiver:
    class Expecting(str, Enum):
        NOTHING = "nothing"
        QUESTION = "question"
        FEEDBACK = "feedback"

    expecting: Expecting
    websocket: WebSocket | None
    queue: asyncio.Queue[InteractionRequest | UserToAgentInteraction | Shutdown]

    def __init__(self, websocket: WebSocket | None):
        self.expecting = WebsocketReceiver.Expecting.QUESTION
        self.websocket = websocket
        self.queue = asyncio.Queue(maxsize=1)

    async def receive_question(self) -> InteractionRequest:
        self.expecting = WebsocketReceiver.Expecting.QUESTION
        msg = await self.queue.get()
        if isinstance(msg, Shutdown):
            raise WebSocketDisconnect()
        if not isinstance(msg, InteractionRequest):
            raise ValueError(f"Expected question but got {type(msg).__name__}")
        return msg

    async def receive_feedback(self) -> UserToAgentInteraction:
        self.expecting = WebsocketReceiver.Expecting.FEEDBACK
        msg = await self.queue.get()
        if isinstance(msg, Shutdown):
            raise WebSocketDisconnect()
        if not isinstance(msg, UserToAgentInteraction):
            raise ValueError(f"Expected feedback but got {type(msg).__name__}")
        return msg

    async def run(self):
        if self.websocket is None:
            raise ValueError("No websocket provided")
        try:
            async for message in self.websocket.iter_json():
                match self.expecting:
                    case WebsocketReceiver.Expecting.NOTHING:
                        await self.websocket.send_json(
                            AragAnswer(
                                exception=ARAGException(
                                    detail="Unexpected message from user"
                                ),
                                operation=AnswerOperation.ERROR,
                            ).model_dump()
                        )
                    case WebsocketReceiver.Expecting.QUESTION:
                        try:
                            await self.queue.put(
                                InteractionRequest.model_validate(message)
                            )
                            self.expecting = WebsocketReceiver.Expecting.NOTHING
                        except ValidationError as e:
                            await self.websocket.send_json(
                                AragAnswer(
                                    exception=ARAGException(
                                        detail="Invalid request payload",
                                        extra={"validation_errors": e.errors()},
                                    ),
                                    operation=AnswerOperation.ERROR,
                                ).model_dump()
                            )
                    case WebsocketReceiver.Expecting.FEEDBACK:
                        try:
                            await self.queue.put(
                                UserToAgentInteraction.model_validate(message)
                            )
                            self.expecting = WebsocketReceiver.Expecting.NOTHING
                        except ValidationError as e:
                            await self.websocket.send_json(
                                AragAnswer(
                                    exception=ARAGException(
                                        detail="Invalid request payload",
                                        extra={"validation_errors": e.errors()},
                                    ),
                                    operation=AnswerOperation.ERROR,
                                ).model_dump()
                            )
        finally:
            await self.queue.put(Shutdown())


async def stream_response(
    app: "HTTPApplication",
    websocket: Optional[WebsocketReceiver],
    account: str,
    agent_id: str,
    session: str,
    interaction: InteractionRequest,
    workflow_id: str = "default",
) -> AsyncIterator[AragAnswer]:
    settings: Settings = app.settings
    agent_manager: AgentManager = app.agent_manager

    try:
        await agent_manager.ensure_workflow_active(account, agent_id, workflow_id)
    except exceptions.NotFoundError as exc:
        yield AragAnswer(
            exception=ARAGException(detail=str(exc)),
            operation=AnswerOperation.ERROR,
        )
        return

    question_id = uuid.uuid4().hex
    subject = settings.answers_subject.format(
        account=account,
        agent_id=agent_id,
        session=session,
        question=question_id,
        workflow_id=workflow_id,
    )

    request = StartInteraction(
        account=account,
        agent_id=agent_id,
        session=session,
        question_id=question_id,
        question=interaction.question,
        headers=interaction.headers,
        arguments=interaction.arguments,
        chat_history=interaction.chat_history,
        workflow_id=workflow_id,
        streaming=interaction.streaming,
    )

    with tracer().start_as_current_span("Request activation"):
        trace_headers: dict[str, str] = {}
        opentelemetry.propagate.inject(trace_headers)
        await app.broker.publish_activation(request, trace_headers)
    try:
        async for _cursor, obj in app.broker.subscribe(subject):
            if isinstance(obj, AgentAnswer):
                yield obj.answer
            elif isinstance(obj, OAuthRequest):
                if websocket is None:
                    yield AragAnswer(
                        exception=ARAGException(
                            detail="Agent requires OAuth which is only supported via websocket"
                        ),
                        operation=AnswerOperation.ERROR,
                    )
                    return

                yield AragAnswer(
                    operation=AnswerOperation.AGENT_REQUEST, oauth=obj.oauth
                )
            elif isinstance(obj, AgentToUserRequest):
                if websocket is None:
                    yield AragAnswer(
                        exception=ARAGException(
                            detail="Agent requires elicitation which is only supported via websocket"
                        ),
                        operation=AnswerOperation.ERROR,
                    )
                    return

                yield AragAnswer(
                    operation=AnswerOperation.AGENT_REQUEST, feedback=obj.feedback
                )
                try:
                    user_response = await websocket.receive_feedback()
                except ValueError as e:
                    # Wrong message type received
                    yield AragAnswer(
                        exception=ARAGException(detail=f"Unexpected message: {str(e)}"),
                        operation=AnswerOperation.ERROR,
                    )
                    return
                except WebSocketDisconnect:
                    return
                await app.broker.send_reply(
                    obj.feedback.feedback_id, user_response.model_dump_json()
                )
            elif isinstance(obj, AgentDone):
                yield AragAnswer(operation=AnswerOperation.DONE)
                break
            elif isinstance(obj, AgentPing):
                pass
            elif isinstance(obj, UserToAgentInteraction):
                # TODO: Stream the user response for full history when resuming
                pass
            else:
                yield AragAnswer(
                    exception=ARAGException(detail="Unknown message from agent"),
                    operation=AnswerOperation.ERROR,
                )
                raise Exception("Unknown message from agent")
    except AgentTimeoutError:
        yield AragAnswer(
            exception=ARAGException(
                detail="Agent has stopped responding. Please, try again."
            ),
            operation=AnswerOperation.ERROR,
        )


async def close_websocket_with_error(
    websocket: WebSocket, task: asyncio.Task, error_message: str
):
    """Send error message and close websocket connection."""
    try:
        await websocket.send_json(
            AragAnswer(
                exception=ARAGException(detail=error_message),
                operation=AnswerOperation.ERROR,
            ).model_dump()
        )
    except (RuntimeError, WebSocketDisconnect):
        pass
    finally:
        task.cancel()
        try:
            await websocket.close()
        except RuntimeError:
            pass


@router.websocket("/api/v1/agent/{agent_id}/session/{session}/ws")
@router.websocket(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/session/{session}/ws"
)
@requires_one([AgentRole.MEMBER])
async def websocket_endpoint(
    websocket: WebSocket,
    agent_id: str,
    session: str,
    keep_open: bool = False,
    workflow_id: str = "default",
    create_session_if_not_exists: bool = Query(True),
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    await websocket.accept()
    receiver = WebsocketReceiver(websocket)
    task = asyncio.create_task(receiver.run())

    # Validate session if agent uses NucliaDB memory (skip for ephemeral sessions)
    agent_manager: AgentManager = websocket.app.agent_manager
    try:
        await agent_manager.ensure_workflow_active(x_stf_account, agent_id, workflow_id)
        if session != "ephemeral" and await agent_has_nucliadb_memory(
            agent_manager, x_stf_account, agent_id, workflow_id
        ):
            ndb: NucliaDBAsync = websocket.app.arag_reader
            session, error_message = await ensure_session_exists(
                ndb, agent_id, session, create_session_if_not_exists
            )
            if error_message:
                await close_websocket_with_error(websocket, task, error_message)
                return
    except Exception as e:
        logger.exception(f"Error checking agent memory config: {e}")
        await close_websocket_with_error(
            websocket, task, f"Failed to verify agent configuration: {str(e)}"
        )
        return

    # Wait for questions
    first_question = True
    while True:
        if not keep_open and not first_question:
            break
        try:
            interaction = await receiver.receive_question()
        except WebSocketDisconnect:
            break
        except ValueError as e:
            # Wrong message type received
            await websocket.send_json(
                AragAnswer(
                    exception=ARAGException(detail=f"Unexpected message: {str(e)}"),
                    operation=AnswerOperation.ERROR,
                ).model_dump()
            )
            break

        first_question = False
        for header, header_value in websocket.headers.items():
            interaction.headers[header] = header_value

        async for msg in stream_response(
            websocket.app,
            receiver,
            x_stf_account,
            agent_id,
            session,
            interaction,
            workflow_id=workflow_id,
        ):
            try:
                await websocket.send_text(msg.model_dump_json())
            except (RuntimeError, WebSocketDisconnect):
                # WebSocket already closed
                pass

    try:
        task.cancel()
        await websocket.close()
    except RuntimeError:
        # WebSocket already closed
        pass


@router.post(
    "/api/v1/agent/{agent_id}/session/{session}",
    status_code=200,
    description="Interact session",
    tags=["Retrieval Agent"],
)
@router.post(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/session/{session}",
    status_code=200,
    description="Interact session",
    tags=["Retrieval Agent"],
)
@requires_one([AgentRole.MEMBER])
async def interaction(
    request: Request,
    agent_id: str,
    session: str,
    item: InteractionRequest,
    workflow_id: str = "default",
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    async def responder():
        async for msg in stream_response(
            request.app,
            None,
            x_stf_account,
            agent_id,
            session,
            item,
            workflow_id=workflow_id,
        ):
            yield msg.model_dump_json() + "\n"

    # subscribe to
    return StreamingResponse(responder(), media_type="application/x-ndjson")
