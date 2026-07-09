"""Session management functions for ARAG agents with NucliaDB memory."""

from typing import Optional
from uuid import UUID

from nucliadb_models import (
    CreateResourcePayload,
    ResourceCreated,
    TextField,
    TextFormat,
    UpdateResourcePayload,
)
from nucliadb_models.conversation import InputConversationField
from nucliadb_models.resource import Resource, ResourceList
from nucliadb_sdk import NucliaDBAsync
from nucliadb_sdk.v2.exceptions import NotFoundError

from hyperforge import logger
from hyperforge.api.models import INFO_FIELD_ID
from hyperforge.memory.memory import QUESTION_ANSWERS_FIELD


async def create_session_resource(
    ndb: NucliaDBAsync,
    agent_id: str,
    slug: str,
    title: str,
    summary: str,
    data: str,
    format: TextFormat = TextFormat.PLAIN,
) -> ResourceCreated:
    """Create a new session resource in the agent's memory KB.

    Args:
        ndb: NucliaDB client instance
        agent_id: The agent/KB ID
        slug: Optional resource slug (ID)
        title: Optional session title
        summary: Optional session summary
        data: Session data text content
        format: Text format (default: PLAIN)

    Returns:
        ResourceCreated with the new session details

    Raises:
        Exception: If session creation fails
    """
    try:
        created = await ndb.create_resource(
            kbid=agent_id,
            content=CreateResourcePayload(
                title=title,
                slug=slug,
                summary=summary,
                texts={INFO_FIELD_ID: TextField(body=data, format=format)},
                conversations={QUESTION_ANSWERS_FIELD: InputConversationField()},
            ),
        )
        logger.info(f"Created session {created.uuid} for agent {agent_id}")
        return created
    except Exception as e:
        logger.exception(f"Error creating session for agent {agent_id}: {e}")
        raise


async def get_session_resource(
    ndb: NucliaDBAsync,
    agent_id: str,
    session_id: str,
    show: Optional[list[str]] = None,
) -> Resource:
    """Get a session resource by ID.

    Args:
        ndb: NucliaDB client instance
        agent_id: The agent/KB ID
        session_id: The session resource ID
        show: Optional list of fields to include (e.g., ["basic", "values"])

    Returns:
        Resource object

    Raises:
        NotFoundError: If session doesn't exist
    """
    query_params = {}
    if show:
        query_params["show"] = show

    return await ndb.get_resource_by_id(
        rid=session_id,
        kbid=agent_id,
        query_params=query_params,
    )


async def session_exists(
    ndb: NucliaDBAsync,
    agent_id: str,
    session_id: str,
) -> bool:
    """Check if a session resource exists.

    Args:
        ndb: NucliaDB client instance
        agent_id: The agent/KB ID
        session_id: The session resource ID

    Returns:
        True if session exists, False otherwise
    """
    try:
        UUID(session_id)
    except ValueError:
        return False

    try:
        await ndb.get_resource_by_id(
            rid=session_id,
            kbid=agent_id,
            query_params={"show": ["basic"]},
        )
        return True
    except NotFoundError:
        return False


async def update_session_resource(
    ndb: NucliaDBAsync,
    agent_id: str,
    session_id: str,
    title: str,
    summary: str,
    data: str,
    format: TextFormat,
) -> None:
    """Update a session resource.

    Args:
        ndb: NucliaDB client instance
        agent_id: The agent/KB ID
        session_id: The session resource ID
        title: Optional new title
        summary: Optional new summary
        data: Optional new data text content
        format: Optional text format

    Raises:
        NotFoundError: If session doesn't exist
    """
    await ndb.update_resource(
        rid=session_id,
        kbid=agent_id,
        content=UpdateResourcePayload(
            title=title,
            summary=summary,
            texts={INFO_FIELD_ID: TextField(body=data, format=format)},
        ),
    )


async def delete_session_resource(
    ndb: NucliaDBAsync,
    agent_id: str,
    session_id: str,
) -> None:
    """Delete a session resource.

    Args:
        ndb: NucliaDB client instance
        agent_id: The agent/KB ID
        session_id: The session resource ID

    Raises:
        NotFoundError: If session doesn't exist
    """
    await ndb.delete_resource(
        rid=session_id,
        kbid=agent_id,
    )


async def list_session_resources(
    ndb: NucliaDBAsync,
    agent_id: str,
    page: int = 0,
    size: int = 20,
) -> ResourceList:
    """List session resources for an agent.

    Args:
        ndb: NucliaDB client instance
        agent_id: The agent/KB ID
        page: Page number (0-based)
        size: Page size

    Returns:
        ResourceList with sessions
    """
    return await ndb.list_resources(
        kbid=agent_id, query_params={"page": page, "size": size}
    )
