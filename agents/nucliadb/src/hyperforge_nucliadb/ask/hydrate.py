import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

from hyperforge import logger
from hyperforge.models import Chunk, Context, FieldTypes
from nuclia.lib.nua_responses import Image
from nucliadb_models import PagePositions
from nucliadb_models.common import FieldTypeName, Paragraph
from nucliadb_models.graph.responses import (
    GraphNodesSearchResponse,
    GraphRelationsSearchResponse,
    GraphSearchResponse,
)
from nucliadb_models.hydration import (
    Hydrated,
    HydrateRequest,
    Hydration,
    ImageParagraphHydration,
    ParagraphHydration,
    ParagraphPageHydration,
    TableParagraphHydration,
)
from nucliadb_models.resource import FileFieldExtractedData, LinkFieldExtractedData
from nucliadb_models.search import KnowledgeboxFindResults

from hyperforge_nucliadb.ask.models import NDBChunk, SearchResults
from hyperforge_nucliadb.driver import NucliaDBDriver


class Relation(str, Enum):
    NEXT = "next"
    PREV = "prev"
    PARENT = "parent"
    REPLACEMENT = "replacement"
    SIBLING = "sibling"


@dataclass
class HydrateData:
    chunks: Dict[str, NDBChunk] = field(default_factory=dict)
    paragraphs: Dict[str, Paragraph] = field(default_factory=dict)
    links: Dict[Tuple[str, str], str] = field(default_factory=dict)
    pages: Dict[int, PagePositions] = field(default_factory=dict)
    images: List[str] = field(default_factory=list)
    pages_image: List[str] = field(default_factory=list)


def generate_chunk(retrieval: KnowledgeboxFindResults, data: HydrateData, kbid: str):
    if retrieval.resources is None:
        return
    for (
        resource_id,
        resource_obj,
    ) in retrieval.resources.items():
        resource_labels: List[str] = []
        if resource_obj.usermetadata:
            resource_labels = [
                f"l/{label.labelset}/{label.label}"
                for label in resource_obj.usermetadata.classifications
            ]

        for field_id, field_obj in resource_obj.fields.items():
            chunk_data = field_id.split("/")  # /t/page
            field_type = chunk_data[1]
            field_name = chunk_data[2]

            # split = field_obj.split  # TODO
            if (
                field_type == FieldTypeName.CONVERSATION.abbreviation()
                and resource_obj.data
                and resource_obj.data.conversations
            ):
                field: FieldTypes = resource_obj.data.conversations[field_name]
            elif (
                field_type == FieldTypeName.FILE.abbreviation()
                and resource_obj.data
                and resource_obj.data.files
            ):
                field = resource_obj.data.files[field_name]
            elif (
                field_type == FieldTypeName.TEXT.abbreviation()
                and resource_obj.data
                and resource_obj.data.texts
            ):
                field = resource_obj.data.texts[field_name]
            elif (
                field_type == FieldTypeName.GENERIC.abbreviation()
                and resource_obj.data
                and resource_obj.data.generics
            ):
                field = resource_obj.data.generics[field_name]
            elif (
                field_type == FieldTypeName.LINK.abbreviation()
                and resource_obj.data
                and resource_obj.data.links
            ):
                field = resource_obj.data.links[field_name]
            else:
                raise Exception(
                    f"Unknown field type {field_type} for field {field_name} in resource {resource_id}"
                )

            if (
                isinstance(field.extracted, FileFieldExtractedData)
                and field.extracted.file is not None
                and field.extracted.file.file_pages_previews is not None
                and field.extracted.file.file_pages_previews.positions is not None
            ):
                data.pages = {}
                for page_num, page_obj in enumerate(
                    field.extracted.file.file_pages_previews.positions
                ):
                    data.pages_image.append(
                        f"/v1/kb/{kbid}/resource/{resource_id}/{field_type}/{field_name}/download/extracted/extracted_images_{page_num}.png"
                    )
                    data.pages[page_num] = page_obj

            if (
                isinstance(field.extracted, LinkFieldExtractedData)
                and field.extracted.link is not None
                and field.extracted.link.link_image is not None
            ):
                data.images.append(
                    f"/v1/kb/{kbid}/resource/{resource_id}/{field_type}/{field_name}/download/extracted/link_thumbnail"
                )

            if field.extracted is not None and field.extracted.metadata is not None:
                previous_paragraph = None
                for paragraph in field.extracted.metadata.metadata.paragraphs:
                    actual_paragraph = (
                        f"{resource_id}/{field_id}/{paragraph.start}-{paragraph.end}"
                    )
                    if previous_paragraph is not None:
                        data.links[(previous_paragraph, Relation.NEXT)] = (
                            actual_paragraph
                        )
                        data.links[(actual_paragraph, Relation.PREV)] = (
                            previous_paragraph
                        )

                    if paragraph.relations is not None:
                        for parent in paragraph.relations.parents:
                            data.links[(actual_paragraph, Relation.PARENT)] = parent
                        for replacement in paragraph.relations.replacements:
                            data.links[(actual_paragraph, Relation.REPLACEMENT)] = (
                                replacement
                            )
                        for sibling in paragraph.relations.siblings:
                            data.links[(actual_paragraph, Relation.SIBLING)] = sibling

                    data.paragraphs[actual_paragraph] = paragraph
                    previous_paragraph = actual_paragraph

            for chunk_id, chunk_obj in field_obj.paragraphs.items():
                data.chunks[chunk_id] = NDBChunk(
                    chunk_id=chunk_id,
                    text=chunk_obj.text,
                    page_with_visual=chunk_obj.page_with_visual,
                    reference=chunk_obj.reference,
                    start=(
                        chunk_obj.position.start
                        if chunk_obj.position is not None
                        else None
                    ),
                    end=(
                        chunk_obj.position.end
                        if chunk_obj.position is not None
                        else None
                    ),
                    page=(
                        chunk_obj.position.page_number
                        if chunk_obj.position is not None
                        else None
                    ),
                    labels=chunk_obj.labels if chunk_obj.labels is not None else [],
                    resource_labels=resource_labels,
                    field=field,
                    link=resource_obj.origin.url or resource_obj.origin.path
                    if resource_obj.origin
                    else None,
                )


def generate_graph_relations_chunk(
    retrieval: GraphRelationsSearchResponse, data: HydrateData
):
    chunk_id = f"relations-{uuid.uuid4().hex}"
    relation_chunk = "Existing relations that can be used to query:\n"
    for relation in retrieval.relations:
        relation_chunk += f"- {relation.type.value}: {relation.label} \n"

    data.chunks[chunk_id] = NDBChunk(
        chunk_id=chunk_id,
        text=relation_chunk,
        page_with_visual=False,
        reference=None,
        start=None,
        end=None,
        page=None,
        labels=[],
        resource_labels=[],
        field=None,
        link=None,
    )


def generate_graph_search_chunk(retrieval: GraphSearchResponse, data: HydrateData):
    chunk_id = f"paths-{uuid.uuid4().hex}"
    relation_chunk = "Existing nodes and relations:\n"
    for path in retrieval.paths:
        relation_chunk += f"- Source node: {path.source.value} {path.source.type.value} {path.source.group}\n"
        relation_chunk += (
            f"  Relation: {path.relation.label} {path.relation.type.value}\n"
        )
        relation_chunk += f"  Destination node: {path.destination.value} {path.destination.type.value} {path.destination.group}\n"

    data.chunks[chunk_id] = NDBChunk(
        chunk_id=chunk_id,
        text=relation_chunk,
        page_with_visual=False,
        reference=None,
        start=None,
        end=None,
        page=None,
        labels=[],
        resource_labels=[],
        field=None,
        link=None,
    )


def generate_graph_nodes_chunk(retrieval: GraphNodesSearchResponse, data: HydrateData):
    chunk_id = f"nodes-{uuid.uuid4().hex}"
    relation_chunk = "Existing Nodes that can be used at query time:\n"
    for node in retrieval.nodes:
        relation_chunk += f"- {node.group}: {node.type.value} - {node.value} \n"

    data.chunks[chunk_id] = NDBChunk(
        chunk_id=chunk_id,
        text=relation_chunk,
        page_with_visual=False,
        reference=None,
        start=None,
        end=None,
        page=None,
        labels=[],
        resource_labels=[],
        field=None,
        link=None,
    )


async def hydrate_images(
    chunk_ids: list[str],
    context: Context,
    nucliadb_driver: NucliaDBDriver,
    vllm: bool,
    visual: bool,
) -> None:
    """Image strategies

    When using a vLLM, we can leverage images into the context. We can get
    images from OCR and inception paragraphs, as well as table images and page
    previews.

    """
    if not vllm:
        return

    resp = await nucliadb_driver.driver.session.post(
        f"/v1/kb/{nucliadb_driver.config.kbid}/hydrate",
        json=HydrateRequest(
            data=chunk_ids,
            hydration=Hydration(
                paragraph=ParagraphHydration(
                    text=False,
                    image=ImageParagraphHydration(
                        source_image=(vllm and visual),
                    ),
                    table=TableParagraphHydration(
                        table_page_preview=vllm,
                    ),
                    page=ParagraphPageHydration(
                        page_with_visual=(vllm and visual),
                    ),
                )
            ),
        ).model_dump(),
    )
    if resp.status_code != 200:
        logger.warning(
            f"Hydration API didn't succeed: {resp.status_code} {resp.content!r}"
        )
        return

    hydrated = Hydrated.model_validate(resp.json())

    for chunk_id, hydrated_paragraph in hydrated.paragraphs.items():
        if (
            hydrated_paragraph.image is not None
            and hydrated_paragraph.image.source_image is not None
        ):
            image = hydrated_paragraph.image.source_image
            context.images[chunk_id] = Image(
                content_type=image.content_type,
                b64encoded=image.b64encoded,
            )
            continue

        hydrated_field = hydrated.fields[hydrated_paragraph.field]

        if (
            hydrated_paragraph.page is not None
            and hydrated_paragraph.page.page_preview_ref is not None
            and hasattr(hydrated_field, "previews")
        ):
            image = hydrated_field.previews[hydrated_paragraph.page.page_preview_ref]  # type: ignore
            context.images[chunk_id] = Image(
                content_type=image.content_type,
                b64encoded=image.b64encoded,
            )
            continue

        if (
            hydrated_paragraph.table is not None
            and hydrated_paragraph.table.page_preview_ref is not None
            and hasattr(hydrated_field, "previews")
        ):
            image = hydrated_field.previews[hydrated_paragraph.table.page_preview_ref]  # type: ignore
            context.images[chunk_id] = Image(
                content_type=image.content_type,
                b64encoded=image.b64encoded,
            )
            continue


async def hydrate(
    context: Context,
    search_results: SearchResults,
    nucliadb_driver: NucliaDBDriver,
    vllm: bool = True,
    full_resource: bool = False,
    after: int = 2,
    before: int = 2,
    visual: bool = False,
    link: bool = False,
    module: str = "ask",
):
    data = HydrateData()
    if search_results.semantic_find_results is not None:
        generate_chunk(
            search_results.semantic_find_results, data, kbid=nucliadb_driver.config.kbid
        )
    if search_results.lexical_find_results is not None:
        generate_chunk(
            search_results.lexical_find_results, data, kbid=nucliadb_driver.config.kbid
        )
    if search_results.graph_find_results is not None:
        generate_chunk(
            search_results.graph_find_results, data, kbid=nucliadb_driver.config.kbid
        )

    if search_results.relations_results is not None:
        generate_graph_relations_chunk(search_results.relations_results, data)
    if search_results.graph_results is not None:
        generate_graph_search_chunk(search_results.graph_results, data)
    if search_results.nodes_results is not None:
        generate_graph_nodes_chunk(search_results.nodes_results, data)

    # ASK DETERMINISTIC RULES

    # Image strategies: hydrate images for chunks when appropiate and we can
    await hydrate_images(
        [chunk_id for chunk_id in data.chunks],
        context,
        nucliadb_driver,
        vllm,
        visual,
    )

    # If there is generated fields add then on FieldExtensionStrategy RAG + extra_fields

    # Has Conversations on the KB facets -> Conversational RAG strategy
    for chunk_id, chunk in data.chunks.items():
        if chunk_id.count("/") < 3:
            context.chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk.text,
                    labels=chunk.labels + chunk.resource_labels,
                    source=nucliadb_driver.config.kbid,
                    origin_agent=module,
                )
            )
        else:
            # Its a field chunk
            field_parts = chunk_id.split("/")
            if len(field_parts) == 4:
                rid, field_type, field_id, start_stop = field_parts
                split = None
            else:
                rid, field_type, field_id, start_stop, split = field_parts

            # If its conversational field get more info
            # if field_type == "c":

            #     if vllm:
            #         attachments_images = True
            #         attachments_text = False
            #     else:
            #         attachments_images = False
            #         attachments_text = True

            # Cut the proper text of the chunk
            if (
                chunk.field is not None
                and chunk.field.extracted is not None
                and chunk.field.extracted.text is not None
                and chunk.field.extracted.text.text is not None
            ):
                text = ""
                if full_resource:
                    text += chunk.field.extracted.text.text
                else:
                    positions: List[Tuple[int, int]] = cut_the_proper_text(
                        chunk_id, chunk, before, after, data
                    )

                    for position in positions:
                        start, end = position
                        text += chunk.field.extracted.text.text[start:end]
                        text += "\n"
                if chunk.link is not None and link:
                    text += f"\n\nLink: {chunk.link}\n"
            else:
                text = chunk.text

            context.chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    labels=chunk.labels + chunk.resource_labels,
                    source=nucliadb_driver.config.kbid,
                    origin_agent=module,
                )
            )


def cut_the_proper_text(
    chunk_id: str,
    chunk: NDBChunk,
    before: int,
    after: int,
    data: HydrateData,
):
    positions_to_cut: List[Tuple[int, int]] = []
    # If add paragraphs
    # Add before and after paragraphs
    start = chunk.start or 0
    end = chunk.end or 0
    actual_id = chunk_id

    # Before and after paragraphs
    for _ in range(before):
        prev_chunk = data.links.get((actual_id, Relation.PREV))
        if prev_chunk is not None:
            new_paragraph = data.paragraphs.get(prev_chunk)
            if new_paragraph is not None:
                actual_id = prev_chunk
                if new_paragraph.start is not None and new_paragraph.start < start:
                    start = new_paragraph.start

    for _ in range(after):
        next_chunk = data.links.get((chunk_id, Relation.NEXT))
        if next_chunk is not None:
            new_paragraph = data.paragraphs.get(next_chunk)
            if new_paragraph is not None:
                actual_id = next_chunk
                if new_paragraph.end is not None and new_paragraph.end > end:
                    end = new_paragraph.end
    positions_to_cut.append((start, end))

    return positions_to_cut
