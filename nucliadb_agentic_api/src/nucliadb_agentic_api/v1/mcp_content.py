"""MCP content conversion utilities.

Converts AragAnswer and generic Python objects into MCP content blocks
(TextContent, ImageContent, EmbeddedResource) suitable for returning from
tool calls.
"""

import json
from itertools import chain
from typing import Any, Sequence

import pydantic_core
from hyperforge.interaction import AragAnswer
from hyperforge.models import Answer, Chunk
from mcp.server.fastmcp.utilities.types import Image
from mcp.types import (
    Annotations,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
)


def _answer_obj_to_content(
    answer: Answer,
) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Convert a hyperforge Answer object to MCP content blocks.

    Handles all fields that Answer can carry: text, citations, chunks,
    structured items, images, image_urls, and visualizations.
    """
    contents: list[TextContent | ImageContent | EmbeddedResource] = []

    if answer.answer:
        contents.append(TextContent(type="text", text=answer.answer))

    if answer.citations and answer.citations.metadata:
        citations_json = json.dumps(pydantic_core.to_jsonable_python(answer.citations))
        contents.append(
            EmbeddedResource(
                type="resource",
                resource=TextResourceContents(
                    uri="rao-response://answer/citations",  # type: ignore[arg-type]
                    text=citations_json,
                    mimeType="application/json",
                ),
            )
        )

    if answer.chunks:
        for idx, chunk in enumerate(answer.chunks):
            if chunk:
                # Extract text from Chunk object or use string directly
                chunk_text = chunk.text if isinstance(chunk, Chunk) else chunk
                if chunk_text:
                    contents.append(
                        EmbeddedResource(
                            type="resource",
                            resource=TextResourceContents(
                                uri=f"rao-response://answer/chunk/{idx}",  # type: ignore[arg-type]
                                text=chunk_text,
                                mimeType="text/plain",
                            ),
                        )
                    )

    if answer.structured:
        for idx, item in enumerate(answer.structured):
            if item:
                contents.append(
                    EmbeddedResource(
                        type="resource",
                        resource=TextResourceContents(
                            uri=f"rao-response://answer/structured/{idx}",  # type: ignore[arg-type]
                            text=item,
                            mimeType="text/plain",
                        ),
                    )
                )

    if answer.images:
        for image in answer.images.values():
            contents.append(
                ImageContent(
                    type="image",
                    data=image.b64encoded,
                    mimeType=image.content_type,
                )
            )

    if answer.image_urls:
        contents.append(
            EmbeddedResource(
                type="resource",
                resource=TextResourceContents(
                    uri="rao-response://answer/image-urls",  # type: ignore[arg-type]
                    text=json.dumps(answer.image_urls),
                    mimeType="application/json",
                ),
            )
        )

    if answer.data_visualizations:
        for idx, viz in enumerate(answer.data_visualizations):
            contents.append(
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri=f"rao-response://answer/visualization/{idx}",  # type: ignore[arg-type]
                        text=json.dumps(
                            pydantic_core.to_jsonable_python(viz.vega_lite_obj)
                        ),
                        mimeType="application/vnd.vegalite.v5+json",
                    ),
                )
            )

    return contents


def convert_arag_answer_to_content(
    msg: AragAnswer,
) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """Convert an AragAnswer to a sequence of MCP content objects.

    Parses **every** populated field independently — the same AragAnswer
    message never carries all fields at once, but any combination is safe.

    Emits (in order):
    - possible_answer content (text, citations, chunks, structured, images,
      image_urls, visualizations) via _answer_obj_to_content
    - TextContent for the main answer text
    - EmbeddedResource (application/json) for top-level citations
    - EmbeddedResource (application/json) for answer URLs
    - EmbeddedResource (text/plain) per context chunk, with metadata
    - ImageContent per base64 image in the context
    - EmbeddedResource (text/plain) per structured item in the context
    - EmbeddedResource (application/json) for context image URLs
    - EmbeddedResource (application/vnd.vegalite.v5+json) per visualization
    - TextContent (assistant-only annotation) for the step, if present
    - TextContent for exception detail, if present
    """
    contents: list[TextContent | ImageContent | EmbeddedResource] = []

    # --- Possible answer (emitted by add_answer callbacks during generation) ---
    if msg.possible_answer:
        contents.extend(_answer_obj_to_content(msg.possible_answer))

    # --- Main answer text (emitted by send_final_answer / session.py) ---
    if msg.answer:
        contents.append(TextContent(type="text", text=msg.answer))

    # --- Top-level citations (from send_final_answer / session.py) ---
    if msg.answer_citations and msg.answer_citations.metadata:
        citations_json = json.dumps(
            pydantic_core.to_jsonable_python(msg.answer_citations)
        )
        contents.append(
            EmbeddedResource(
                type="resource",
                resource=TextResourceContents(
                    uri="rao-response://citations",  # type: ignore[arg-type]
                    text=citations_json,
                    mimeType="application/json",
                ),
            )
        )

    # --- Answer URLs ---
    if msg.answer_urls:
        contents.append(
            EmbeddedResource(
                type="resource",
                resource=TextResourceContents(
                    uri="rao-response://answer-urls",  # type: ignore[arg-type]
                    text=json.dumps(msg.answer_urls),
                    mimeType="application/json",
                ),
            )
        )

    # --- Context chunks and images (emitted by save_context callbacks) ---
    if msg.context:
        context = msg.context
        for chunk in context.chunks:
            meta: dict[str, Any] = {}
            if chunk.title:
                meta["title"] = chunk.title
            if chunk.source:
                meta["source"] = chunk.source
            if chunk.labels:
                meta["labels"] = chunk.labels
            if chunk.origin_url:
                meta["origin_url"] = chunk.origin_url
            if chunk.url:
                meta["url"] = chunk.url
            contents.append(
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri=f"rao-response://context/{context.id}/chunk/{chunk.chunk_id}",  # type: ignore[arg-type]
                        text=chunk.text,
                        mimeType="text/plain",
                        **{"_meta": meta} if meta else {},
                    ),
                )
            )

        for image in context.images.values():
            contents.append(
                ImageContent(
                    type="image",
                    data=image.b64encoded,
                    mimeType=image.content_type,
                )
            )

        for idx, structured in enumerate(context.structured):
            contents.append(
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri=f"rao-response://context/{context.id}/structured/{idx}",  # type: ignore[arg-type]
                        text=structured,
                        mimeType="text/plain",
                    ),
                )
            )

        if context.image_urls:
            contents.append(
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri=f"rao-response://context/{context.id}/image-urls",  # type: ignore[arg-type]
                        text=json.dumps(context.image_urls),
                        mimeType="application/json",
                    ),
                )
            )

    # --- Top-level visualizations (from session.py final answer) ---
    if msg.data_visualizations:
        for idx, viz in enumerate(msg.data_visualizations):
            contents.append(
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri=f"rao-response://visualization/{idx}",  # type: ignore[arg-type]
                        text=json.dumps(
                            pydantic_core.to_jsonable_python(viz.vega_lite_obj)
                        ),
                        mimeType="application/vnd.vegalite.v5+json",
                    ),
                )
            )

    # --- Step (assistant-only, not shown to user) ---
    if msg.step:
        step = msg.step
        parts = [f"Step: {step.title}"]
        if step.value:
            parts.append(f"Value: {step.value}")
        if step.reason:
            parts.append(f"Reason: {step.reason}")
        if step.error:
            parts.append(f"Error: {step.error}")
        contents.append(
            TextContent(
                type="text",
                text="\n".join(parts),
                annotations=Annotations(audience=["assistant"]),
            )
        )

    # --- Exception ---
    if msg.exception:
        contents.append(TextContent(type="text", text=f"Error: {msg.exception.detail}"))

    return contents


def convert_to_content(
    result: Any,
) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """Convert a generic result to a sequence of MCP content objects.

    Handles MCP content types directly, Image objects, lists/tuples
    (recursively), and serialises everything else as JSON text.
    """
    if result is None:
        return []

    if isinstance(result, (TextContent, ImageContent, EmbeddedResource)):
        return [result]

    if isinstance(result, Image):
        return [result.to_image_content()]

    if isinstance(result, (list, tuple)):
        return list(chain.from_iterable(convert_to_content(item) for item in result))

    if not isinstance(result, str):
        try:
            result = json.dumps(pydantic_core.to_jsonable_python(result))
        except Exception:
            result = str(result)

    return [TextContent(type="text", text=result)]
