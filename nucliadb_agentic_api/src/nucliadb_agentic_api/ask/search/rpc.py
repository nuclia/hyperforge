import base64
from typing import Literal

from nucliadb_models.augment import AugmentRequest, AugmentResponse
from nucliadb_models.configuration import SearchConfiguration
from nucliadb_models.graph.requests import GraphNodesSearchRequest, GraphSearchRequest
from nucliadb_models.graph.responses import (
    GraphNodesSearchResponse,
    GraphSearchResponse,
)
from nucliadb_models.labels import KnowledgeBoxLabels
from nucliadb_models.retrieval import RetrievalRequest, RetrievalResponse
from nucliadb_models.search import (
    FindRequest,
    Image,
    KnowledgeboxFindResults,
    NucliaDBClientType,
)
from nucliadb_sdk import NucliaDBAsync
from nucliadb_sdk.v2.exceptions import NotFoundError, UnknownError
from pydantic import TypeAdapter
from typing_extensions import assert_never

from nucliadb_agentic_api.ask import logger
from nucliadb_agentic_api.ask.exceptions import (
    KnowledgeBoxNotFound,
    NucliaDBError,
)
from nucliadb_agentic_api.ask.settings import settings
from nucliadb_agentic_api.ask.utils.ids import FieldId


async def get_resource_uuid_from_slug(
    reader_sdk: NucliaDBAsync, kbid: str, slug: str
) -> str | None:
    try:
        resource = await reader_sdk.get_resource_by_slug(kbid=kbid, slug=slug)
    except NotFoundError:
        return None
    else:
        return resource.id


async def get_search_configuration(
    reader_sdk: NucliaDBAsync, kbid: str, name: str
) -> SearchConfiguration | None:
    resp = await reader_sdk.session.get(f"/v1/kb/{kbid}/search_configurations/{name}")

    if resp.status_code == 404:
        return None

    if resp.status_code != 200:
        raise Exception(
            f"/search_configurations/{name} call failed: {resp.status_code} {resp.content.decode()}"
        )

    config: SearchConfiguration = TypeAdapter(SearchConfiguration).validate_json(
        resp.content
    )
    return config


async def find(
    search_sdk: NucliaDBAsync,
    kbid: str,
    item: FindRequest,
    x_ndb_client: NucliaDBClientType,
    x_nucliadb_user: str,
    x_forwarded_for: str,
) -> tuple[KnowledgeboxFindResults, bool]:
    """RPC to /find endpoint making it look as an internal call."""

    resp = await search_sdk.session.post(
        f"/v1/kb/{kbid}/find",
        headers={
            "x-ndb-client": x_ndb_client,
            "x-nucliadb-user": x_nucliadb_user,
            "x-forwarded-for": x_forwarded_for,
        },
        json=item.model_dump(),
    )
    if resp.status_code == 200:
        incomplete = False
    elif resp.status_code == 206:
        incomplete = True
    elif resp.status_code == 404:
        raise KnowledgeBoxNotFound()
    else:
        raise Exception(
            f"/find call failed: {resp.status_code} {resp.content.decode()}"
        )

    find_results = KnowledgeboxFindResults.model_validate(resp.json())

    return find_results, incomplete


async def retrieve(
    search_sdk: NucliaDBAsync,
    kbid: str,
    item: RetrievalRequest,
) -> RetrievalResponse:
    try:
        retrieved = await search_sdk.retrieve(kbid=kbid, content=item)
    except NotFoundError as exc:
        raise KnowledgeBoxNotFound() from exc
    except UnknownError as exc:
        logger.warning(
            "/retrieve RPC to NucliaDB failed", extra={"kbid": kbid}, exc_info=True
        )
        raise NucliaDBError() from exc
    return retrieved


async def augment(
    search_sdk: NucliaDBAsync, kbid: str, item: AugmentRequest
) -> AugmentResponse:
    try:
        augmented = await search_sdk.augment(kbid=kbid, content=item)
    except NotFoundError as exc:
        raise KnowledgeBoxNotFound() from exc
    except UnknownError as exc:
        logger.warning(
            "/augment RPC to NucliaDB failed", extra={"kbid": kbid}, exc_info=True
        )
        raise NucliaDBError() from exc
    return augmented


async def graph_paths(
    search_sdk: NucliaDBAsync, kbid: str, item: GraphSearchRequest
) -> GraphSearchResponse:
    paths = await search_sdk.graph_search(kbid=kbid, content=item)
    return paths


async def graph_nodes(
    search_sdk: NucliaDBAsync, kbid: str, item: GraphNodesSearchRequest
) -> GraphNodesSearchResponse:
    nodes = await search_sdk.graph_nodes(kbid=kbid, content=item)
    return nodes


async def labelsets(reader_sdk: NucliaDBAsync, kbid: str) -> KnowledgeBoxLabels:
    labelsets = await reader_sdk.get_labelsets(kbid=kbid)
    return labelsets


async def download_image(
    reader_sdk: NucliaDBAsync,
    kbid: str,
    field_id: FieldId,
    path: str,
    *,
    mime_type: str,
) -> Image | None:
    async with (
        reader_sdk.session.stream(
            "GET",
            f"/v1/kb/{kbid}/resource/{field_id.rid}/{field_id.type_name.value}/{field_id.key}/download/extracted/{path}",
        ) as resp,
    ):
        if resp.status_code == 404:
            return None

        data = await resp.aread()
        return Image(
            b64encoded=base64.b64encode(data).decode(),
            content_type=mime_type,
        )


__SDK: dict[Literal["reader"] | Literal["search"], NucliaDBAsync] = {}


def get_sdk(service: Literal["reader"] | Literal["search"]) -> NucliaDBAsync:
    if service in __SDK:
        return __SDK[service]

    if service == "reader":
        service_address = settings.nucliadb_reader_address
    elif service == "search":
        service_address = settings.nucliadb_search_address
    else:
        assert_never(service)

    sdk = NucliaDBAsync(
        url=service_address,
        headers={"x-nucliadb-roles": "READER"},
    )
    __SDK[service] = sdk
    return sdk
