from __future__ import annotations

import asyncio
import itertools
from collections.abc import Sequence
from typing import Any, Literal, cast
from uuid import uuid4

from hyperforge.configure import toolkit
from hyperforge.harness_sdk import (
    BaseToolkit,
    HarnessContextType,
    HarnessEventType,
    HarnessSkill,
    HarnessTool,
    ToolCallContext,
    ToolkitConfig,
    tool,
)
from hyperforge.interaction import OAuthFeedbackReturnSchema
from hyperforge.llm_config import LLMConfig, LLMField, llm_defaults
from hyperforge.models import Chunk, Context
from hyperforge.settings import OAuthSettings
from hyperforge.toolkits.common import (
    RetrievalOutput,
    resolve_manager,
    retrieval_context_factory,
)
from nucliadb_models import SyncMetadata
from nucliadb_models.filters import (
    And,
    CatalogFilterExpression,
    FieldFilterExpression,
    FilterExpression,
    Keyword,
    Or,
    Resource,
    ResourceFilterExpression,
)
from nucliadb_models.search import (
    AskRequest,
    CatalogRequest,
    CitationsType,
    FieldExtensionStrategy,
    FindRequest,
    FullResourceStrategy,
    MetadataExtensionStrategy,
    NeighbouringParagraphsStrategy,
    ResourceProperties,
)
from pydantic import BaseModel, Field

from .ask_utils import (
    combine_catalog_filter_expressions,
    combine_filter_expressions,
    get_chunk_text,
    to_field_filter_expression,
    to_resource_filter_expression,
)
from .driver import NucliaDBDriver
from .sync.driver import SyncDriver


def _resolve_ndb_drivers(
    drivers: dict[str, Any], sources: list[str]
) -> list[tuple[str, NucliaDBDriver]]:
    selected_ids = (
        sources
        if sources
        else [
            key
            for key, drv in drivers.items()
            if getattr(drv, "provider", None) in {"nucliadb", "sync"}
        ]
    )
    resolved: list[tuple[str, NucliaDBDriver]] = []
    for source_id in selected_ids:
        drv = drivers.get(source_id)
        if drv is None:
            raise ValueError(f"NucliaDB source {source_id!r} is not configured")
        resolved.append((source_id, cast(NucliaDBDriver, drv)))
    if not resolved:
        raise ValueError("No NucliaDB driver configured for toolkit")
    return resolved


def _merge_field_filters(
    driver: NucliaDBDriver,
    *,
    keyword_filters: list[str] | None = None,
    and_filters: list[str] | None = None,
    or_filters: list[str] | None = None,
    resource_filters: list[str] | None = None,
) -> FilterExpression | None:
    and_operands: list[FieldFilterExpression] = []
    if keyword_filters:
        if len(keyword_filters) == 1:
            and_operands.append(Keyword(word=keyword_filters[0]))
        else:
            and_operands.append(Or(operands=[Keyword(word=w) for w in keyword_filters]))
    if resource_filters is not None:
        and_operands.append(Or(operands=[Resource(id=rid) for rid in resource_filters]))
    for raw_group, op_cls in (
        (driver.config.filters, And),
        (and_filters, And),
        (or_filters, Or),
    ):
        if raw_group:
            parsed = [
                expr
                for item in raw_group
                if (expr := to_field_filter_expression(item)) is not None
            ]
            if len(parsed) == 1:
                and_operands.append(parsed[0])
            elif len(parsed) > 1:
                and_operands.append(op_cls(operands=parsed))

    expressions: list[FilterExpression] = []
    if len(and_operands) == 1:
        expressions.append(FilterExpression(field=and_operands[0]))
    elif len(and_operands) > 1:
        expressions.append(FilterExpression(field=And(operands=and_operands)))
    if driver.config.filter_expression is not None:
        expressions.append(driver.config.filter_expression)
    if not expressions:
        return None
    if len(expressions) == 1:
        return expressions[0]
    return combine_filter_expressions(expressions, operator="and")


class AskInput(BaseModel):
    question: str = Field(
        description="Question to search and answer from the Knowledge Box."
    )
    labels: list[str] = Field(
        default_factory=list,
        description="Optional label filters in `labelset/label` format.",
    )
    keyword_filters: list[str] = Field(
        default_factory=list,
        description="Optional exact keyword filters.",
    )
    full_resource: bool = Field(
        default=False,
        description="Retrieve full matching resources instead of paragraph windows.",
    )


class AdvancedAskInput(BaseModel):
    question: str
    top_k: int | None = Field(default=None, ge=1, le=200)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    resource_filters: list[str] | None = None
    generate_answer: bool = True


class FindInput(BaseModel):
    query: str = Field(
        description="Semantic/keyword query to locate matching resources."
    )
    keyword_filters: list[str] = Field(default_factory=list)
    and_filters: list[str] = Field(default_factory=list)


class FindOutput(BaseModel):
    resources_by_source: dict[str, list[str]]


class TitleSearchInput(BaseModel):
    title: str = Field(description="Document title or title substring.")
    filters: list[str] = Field(
        default_factory=list,
        description="Optional catalog filters such as `/l/labelset/label` or `/icon/application/pdf`.",
    )


class AskByTitleInput(BaseModel):
    title: str = Field(description="Document title to restrict RAG retrieval to.")
    question: str = Field(description="Question to ask on the matching document(s).")


class CatalogInput(BaseModel):
    query: str = Field(default="", description="Optional catalog query string.")
    filters: list[str] = Field(
        default_factory=list,
        description="Catalog filters such as `/l/topic/ai` or `/icon/application/pdf`.",
    )
    page_size: int = Field(default=25, ge=1, le=200)


class CatalogOutput(BaseModel):
    resources_by_source: dict[str, list[dict[str, Any]]]


class LabelsInput(BaseModel):
    labelset: str = Field(
        default="",
        description="Optional specific labelset name; leave empty to list all labelsets.",
    )


class LabelsOutput(BaseModel):
    labels_by_source: dict[str, dict[str, list[str]]]


class ImagesByTitleInput(BaseModel):
    title: str = Field(description="Document title to extract image URLs from.")


class NucliaDBToolkitConfig(ToolkitConfig):
    toolkit: Literal["nucliadb"] = "nucliadb"
    sources: list[str] = Field(default_factory=list)
    generative_model: LLMField = Field(
        default=LLMConfig(model_id=llm_defaults.default),
        title="NucliaDB generative model",
    )


class NucliaDBSyncToolkitConfig(NucliaDBToolkitConfig):
    toolkit: Literal["nucliadb_sync"] = "nucliadb_sync"  # type: ignore[assignment]
    feedback_timeout_ms: int = Field(default=300_000, ge=1000)


@toolkit(
    id="nucliadb",
    title="NucliaDB Knowledge Box Toolkit",
    description="Semantic/keyword RAG, title search, catalog filtering, label discovery, and image extraction on NucliaDB.",
    config_schema=NucliaDBToolkitConfig,
)
class NucliaDBToolkit(BaseToolkit[NucliaDBToolkitConfig]):
    toolkit_id = "nucliadb"

    async def _filter_authorized_resources(
        self,
        call_context: ToolCallContext,
        source_id: str,
        driver: NucliaDBDriver,
        resource_ids: list[str] | None,
        question: str,
    ) -> list[str] | None:
        del call_context, source_id, driver, question
        return resource_ids

    async def _ask_source(
        self,
        call_context: ToolCallContext,
        source_id: str,
        driver: NucliaDBDriver,
        question: str,
        *,
        keyword_filters: list[str] | None = None,
        and_filters: list[str] | None = None,
        or_filters: list[str] | None = None,
        resource_filters: list[str] | None = None,
        full_resource: bool = False,
        top_k: int | None = None,
        min_score: float | None = None,
        generate_answer: bool = True,
    ) -> Context:
        authorized = await self._filter_authorized_resources(
            call_context, source_id, driver, resource_filters, question
        )
        if authorized is not None and len(authorized) == 0:
            return Context(
                agent_id=self.config.id,
                original_question_uuid=None,
                actual_question_uuid=None,
                question=question,
                source=source_id,
                agent=self.toolkit_id,
                title=self.config.title or f"NucliaDB ({source_id})",
            )

        filter_expr = _merge_field_filters(
            driver,
            keyword_filters=keyword_filters,
            and_filters=and_filters,
            or_filters=or_filters,
            resource_filters=authorized,
        )
        rag_strategies = (
            [
                FullResourceStrategy(count=1),
                MetadataExtensionStrategy(types=["classification_labels", "origin"]),
            ]
            if full_resource
            else [
                FieldExtensionStrategy(fields=["a/title", "a/summary"]),
                NeighbouringParagraphsStrategy(before=5, after=5),
                MetadataExtensionStrategy(types=["classification_labels", "origin"]),
            ]
        )
        ask_req = AskRequest(
            query=question,
            show=[ResourceProperties.BASIC, ResourceProperties.ORIGIN],
            citations=CitationsType.LLM_FOOTNOTES,
            generative_model=self.config.generative_model.model_id,
            filter_expression=filter_expr,
            rag_strategies=rag_strategies,
            generate_answer=generate_answer,
        )
        if top_k is not None:
            ask_req.top_k = top_k
        if min_score is not None:
            ask_req.min_score = min_score

        resp = await driver.ask(ask_req, headers={"X-SHOW-CONSUMPTION": "true"})
        ctx = Context(
            agent_id=self.config.id,
            original_question_uuid=None,
            actual_question_uuid=None,
            question=question,
            source=source_id,
            agent=self.toolkit_id,
            title=self.config.title or f"Retrieval on {source_id}",
        )
        chunk_ids = (
            list(resp.citation_footnote_to_context.values())
            if resp.citation_footnote_to_context
            else list(resp.citations.keys())
            if resp.citations
            else list(resp.retrieval_results.best_matches)
        )
        for chunk_id in chunk_ids:
            rid = chunk_id.split("/")[0]
            resource = resp.retrieval_results.resources.get(rid)
            if resource is None:
                continue
            ctx.chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    title=resource.title,
                    text=get_chunk_text(resp, chunk_id),
                    source=source_id,
                    origin_url=resource.origin.url if resource.origin else None,
                    origin_agent=self.toolkit_id,
                )
            )
        if resp.answer and resp.status == "success":
            ctx.summary = resp.answer
        return ctx

    def build_tools(self) -> Sequence[HarnessTool[Any, Any]]:
        cfg = self.config

        @tool(
            name="ask",
            description="Run semantic and keyword RAG retrieval against configured NucliaDB Knowledge Boxes.",
            context_type=HarnessContextType.RETRIEVAL,
            context_factory=retrieval_context_factory,
        )
        async def ask(
            call_context: ToolCallContext, value: AskInput
        ) -> RetrievalOutput:
            manager = resolve_manager(call_context)
            drivers = _resolve_ndb_drivers(manager.drivers, cfg.sources)
            and_filters: list[str] = []
            or_filters: list[str] = []
            by_set: dict[str, list[str]] = {}
            for lbl in value.labels:
                if "/" in lbl:
                    lset, lval = lbl.split("/", 1)
                    by_set.setdefault(lset, []).append(lval)
            for lset, vals in by_set.items():
                if len(vals) == 1:
                    and_filters.append(f"/l/{lset}/{vals[0]}")
                else:
                    or_filters.extend(f"/l/{lset}/{v}" for v in vals)

            contexts = await asyncio.gather(
                *(
                    self._ask_source(
                        call_context,
                        src_id,
                        drv,
                        value.question,
                        keyword_filters=value.keyword_filters,
                        and_filters=and_filters or None,
                        or_filters=or_filters or None,
                        full_resource=value.full_resource,
                    )
                    for src_id, drv in drivers
                )
            )
            return RetrievalOutput(
                contexts=[c.model_dump(mode="json") for c in contexts]
            )

        @tool(
            name="advanced_ask",
            description="Query NucliaDB with explicit top_k, min_score, and resource_filters controls.",
            context_type=HarnessContextType.RETRIEVAL,
            context_factory=retrieval_context_factory,
        )
        async def advanced_ask(
            call_context: ToolCallContext, value: AdvancedAskInput
        ) -> RetrievalOutput:
            manager = resolve_manager(call_context)
            drivers = _resolve_ndb_drivers(manager.drivers, cfg.sources)
            contexts = await asyncio.gather(
                *(
                    self._ask_source(
                        call_context,
                        src_id,
                        drv,
                        value.question,
                        resource_filters=value.resource_filters,
                        top_k=value.top_k,
                        min_score=value.min_score,
                        generate_answer=value.generate_answer,
                    )
                    for src_id, drv in drivers
                )
            )
            return RetrievalOutput(
                contexts=[c.model_dump(mode="json") for c in contexts]
            )

        @tool(
            name="find",
            description="Locate matching resource IDs in NucliaDB without running generative synthesis.",
        )
        async def find(call_context: ToolCallContext, value: FindInput) -> FindOutput:
            manager = resolve_manager(call_context)
            drivers = _resolve_ndb_drivers(manager.drivers, cfg.sources)
            found: dict[str, list[str]] = {}
            for src_id, drv in drivers:
                expr = _merge_field_filters(
                    drv,
                    keyword_filters=value.keyword_filters or None,
                    and_filters=value.and_filters or None,
                )
                res = await drv.find_raw(
                    FindRequest(query=value.query, filter_expression=expr)
                )
                rids = list(res.resources.keys())
                authorized = await self._filter_authorized_resources(
                    call_context, src_id, drv, rids, value.query
                )
                found[src_id] = authorized if authorized is not None else rids
            return FindOutput(resources_by_source=found)

        @tool(
            name="search_by_title",
            description="Search for resource IDs in NucliaDB matching a document title.",
        )
        async def search_by_title(
            call_context: ToolCallContext, value: TitleSearchInput
        ) -> FindOutput:
            manager = resolve_manager(call_context)
            drivers = _resolve_ndb_drivers(manager.drivers, cfg.sources)
            by_source: dict[str, list[str]] = {}
            for src_id, drv in drivers:
                cat_filters: list[ResourceFilterExpression] = [
                    expr
                    for f in value.filters
                    if (expr := to_resource_filter_expression(f)) is not None
                ]
                to_combine: list[CatalogFilterExpression] = []
                if len(cat_filters) == 1:
                    to_combine.append(CatalogFilterExpression(resource=cat_filters[0]))
                elif len(cat_filters) > 1:
                    to_combine.append(
                        CatalogFilterExpression(resource=And(operands=cat_filters))
                    )
                if drv.config.catalog_filter_expression is not None:
                    to_combine.append(drv.config.catalog_filter_expression)
                cat_expr = (
                    None
                    if not to_combine
                    else to_combine[0]
                    if len(to_combine) == 1
                    else combine_catalog_filter_expressions(to_combine, operator="and")
                )
                resp = await drv.catalog_search_raw(
                    CatalogRequest(
                        query=value.title,
                        page_size=25,
                        page_number=0,
                        filter_expression=cat_expr,
                    )
                )
                rids = list(resp.resources.keys())
                authorized = await self._filter_authorized_resources(
                    call_context, src_id, drv, rids, value.title
                )
                by_source[src_id] = authorized if authorized is not None else rids
            return FindOutput(resources_by_source=by_source)

        @tool(
            name="ask_by_title",
            description="Ask a question restricted to documents matching a specific title.",
            context_type=HarnessContextType.RETRIEVAL,
            context_factory=retrieval_context_factory,
        )
        async def ask_by_title(
            call_context: ToolCallContext, value: AskByTitleInput
        ) -> RetrievalOutput:
            matches = await search_by_title(
                call_context, TitleSearchInput(title=value.title)
            )
            manager = resolve_manager(call_context)
            drivers = dict(_resolve_ndb_drivers(manager.drivers, cfg.sources))
            tasks = [
                self._ask_source(
                    call_context,
                    src_id,
                    drivers[src_id],
                    value.question,
                    resource_filters=rids,
                    full_resource=True,
                )
                for src_id, rids in matches.resources_by_source.items()
                if rids and src_id in drivers
            ]
            contexts = await asyncio.gather(*tasks) if tasks else []
            return RetrievalOutput(
                contexts=[c.model_dump(mode="json") for c in contexts]
            )

        @tool(
            name="catalog_search",
            description="List resources in the Knowledge Box matching metadata or label filters.",
        )
        async def catalog_search(
            call_context: ToolCallContext, value: CatalogInput
        ) -> CatalogOutput:
            manager = resolve_manager(call_context)
            drivers = _resolve_ndb_drivers(manager.drivers, cfg.sources)
            out: dict[str, list[dict[str, Any]]] = {}
            for src_id, drv in drivers:
                cat_filters = [
                    expr
                    for f in value.filters
                    if (expr := to_resource_filter_expression(f)) is not None
                ]
                cat_expr = (
                    CatalogFilterExpression(resource=cat_filters[0])
                    if len(cat_filters) == 1
                    else CatalogFilterExpression(resource=And(operands=cat_filters))
                    if len(cat_filters) > 1
                    else None
                )
                resp = await drv.catalog_search_raw(
                    CatalogRequest(
                        query=value.query,
                        page_size=value.page_size,
                        page_number=0,
                        filter_expression=cat_expr,
                    )
                )
                out[src_id] = [
                    {"id": rid, "title": r.title, "icon": r.icon}
                    for rid, r in resp.resources.items()
                ]
            return CatalogOutput(resources_by_source=out)

        @tool(
            name="list_labels",
            description="List available classification labelsets and label values in NucliaDB.",
        )
        async def list_labels(
            call_context: ToolCallContext, value: LabelsInput
        ) -> LabelsOutput:
            manager = resolve_manager(call_context)
            drivers = _resolve_ndb_drivers(manager.drivers, cfg.sources)
            result: dict[str, dict[str, list[str]]] = {}
            for src_id, drv in drivers:
                all_labels = await drv.labels()
                if value.labelset:
                    result[src_id] = {
                        value.labelset: all_labels.get(value.labelset, [])
                    }
                else:
                    result[src_id] = all_labels
            return LabelsOutput(labels_by_source=result)

        @tool(
            name="get_images",
            description="Retrieve extracted image URLs from documents matching a title.",
            context_type=HarnessContextType.RETRIEVAL,
            context_factory=retrieval_context_factory,
        )
        async def get_images(
            call_context: ToolCallContext, value: ImagesByTitleInput
        ) -> RetrievalOutput:
            matches = await search_by_title(
                call_context, TitleSearchInput(title=value.title)
            )
            manager = resolve_manager(call_context)
            drivers = dict(_resolve_ndb_drivers(manager.drivers, cfg.sources))
            contexts: list[dict[str, Any]] = []
            for src_id, rids in matches.resources_by_source.items():
                drv = drivers.get(src_id)
                if drv is None:
                    continue
                for rid in rids:
                    res = await drv.get_resource_by_id(
                        rid=rid,
                        query_params={
                            "show": ["basic", "extracted"],
                            "extracted": ["metadata", "file"],
                        },
                    )
                    if res is None or res.data is None or res.data.files is None:
                        continue
                    urls: list[str] = []
                    for field in res.data.files.values():
                        if (
                            field.extracted
                            and field.extracted.file
                            and field.extracted.file.nested_list_position
                            and field.extracted.file.file_generated
                        ):
                            for img_name in field.extracted.file.nested_list_position:
                                gen = field.extracted.file.file_generated.get(img_name)
                                if gen and gen.uri:
                                    urls.append(gen.uri)
                    if urls:
                        ctx = Context(
                            agent_id=cfg.id,
                            original_question_uuid=None,
                            actual_question_uuid=None,
                            question=f"Images for {value.title}",
                            source=src_id,
                            agent=self.toolkit_id,
                            title=f"Images: {value.title}",
                            image_urls=urls,
                        )
                        contexts.append(ctx.model_dump(mode="json"))
            return RetrievalOutput(contexts=contexts)

        return [
            ask,
            advanced_ask,
            find,
            search_by_title,
            ask_by_title,
            catalog_search,
            list_labels,
            get_images,
        ]

    def build_skills(self) -> Sequence[HarnessSkill]:
        return [
            HarnessSkill(
                id="nucliadb_search_guide",
                name="NucliaDB Knowledge Box Retrieval Guide",
                description=(
                    "Best practices for querying NucliaDB Knowledge Boxes, label filtering, "
                    "title lookups, and catalog search."
                ),
                instructions=(
                    "Follow these guidelines when querying NucliaDB (`{namespace}`):\n"
                    "1. Use `{namespace_prefix}ask` for general semantic + keyword questions.\n"
                    "2. When the user names a specific document title, prefer `{namespace_prefix}ask_by_title` "
                    "or `{namespace_prefix}search_by_title` for exact document targeting.\n"
                    "3. Before filtering by labels (`labels=['labelset/value']` or catalog filters "
                    "`/l/labelset/value`), call `{namespace_prefix}list_labels` to verify exact "
                    "labelset and label names—never guess label names.\n"
                    "4. Use `{namespace_prefix}catalog_search` with `/l/...` or `/icon/application/pdf` "
                    "filters when enumerating documents by metadata."
                ),
                recommended_tools=(
                    "ask",
                    "ask_by_title",
                    "list_labels",
                    "catalog_search",
                ),
                code_examples=(
                    "batches = parallel([call({namespace_prefix}ask, question=q) for q in queries])\n"
                    'contexts = [c for b in batches for c in b["contexts"]]',
                ),
            )
        ]


@toolkit(
    id="nucliadb_sync",
    title="NucliaDB Sync OAuth Toolkit",
    description="NucliaDB retrieval with per-user Sync OAuth credential elicitation and ACL resource validation.",
    config_schema=NucliaDBSyncToolkitConfig,
)
class NucliaDBSyncToolkit(NucliaDBToolkit):
    toolkit_id = "nucliadb_sync"
    oauth_settings: OAuthSettings = OAuthSettings()

    async def _filter_authorized_resources(
        self,
        call_context: ToolCallContext,
        source_id: str,
        driver: NucliaDBDriver,
        resource_ids: list[str] | None,
        question: str,
    ) -> list[str] | None:
        sync_driver = cast(SyncDriver, driver)
        candidates = resource_ids
        if candidates is None:
            find_res = await driver.find_raw(FindRequest(query=question))
            candidates = list(find_res.resources.keys())
        if not candidates:
            return []

        resources = await asyncio.gather(
            *(
                driver.driver.get_resource_by_id(
                    kbid=driver.config.kbid,
                    rid=rid,
                    query_params={"show": [ResourceProperties.ORIGIN.value]},
                )
                for rid in candidates
            )
        )
        public_rids: set[str] = set()
        by_conn: dict[str, list[str]] = {}
        sync_meta: dict[str, SyncMetadata] = {}

        for res_obj in resources:
            origin = res_obj.origin
            if (
                origin is None
                or not origin.source_id
                or not origin.source_id.startswith("sync_config_")
            ):
                if origin is None or origin.sync_metadata is None:
                    public_rids.add(res_obj.id)
                continue
            if origin.sync_metadata is None:
                continue
            sync_cfg_id = origin.source_id.removeprefix("sync_config_")
            by_conn.setdefault(sync_cfg_id, []).append(res_obj.id)
            sync_meta[res_obj.id] = origin.sync_metadata

        if not by_conn:
            return [rid for rid in candidates if rid in public_rids]

        creds_providers = {}
        for sync_cfg_id in list(by_conn):
            if sync_cfg_id not in sync_driver.sync_configs:
                await sync_driver.resolve_sync_config(sync_cfg_id)
            conns = sync_driver.sync_configs.get(sync_cfg_id, [])
            if conns:
                creds_providers[sync_cfg_id] = sync_driver.information[
                    conns[0]
                ].provider

        harness = call_context.harness
        raw_reply = await harness.request_feedback(
            question="Get credentials",
            response_schema=OAuthFeedbackReturnSchema.model_json_schema(),
            timeout_ms=cast(NucliaDBSyncToolkitConfig, self.config).feedback_timeout_ms,
        )
        existing_creds: dict[str, dict[str, str]] = {}
        if raw_reply:
            parsed_reply = OAuthFeedbackReturnSchema.model_validate_json(raw_reply)
            existing_creds = parsed_reply.existing_credentials or {}

        if not existing_creds:
            mem = harness.execution_context.get("memory")
            oauth_uuid = uuid4().hex
            redirect_url = self.oauth_settings.rao_redirect_url.format(
                agent_id=mem.get_agent_id() if mem else "harness",
                session_id=harness.conversation_id,
                oauth_uuid=oauth_uuid,
                workflow_id=mem.get_workflow_id() if mem else "default",
            )
            for sync_cfg_id in creds_providers:
                real_conn = sync_driver.sync_configs[sync_cfg_id][0]
                oauth_url = await sync_driver.get_oauth_url(
                    connection_id=real_conn,
                    rao_redirect_url=redirect_url,
                    oauth_uuid=oauth_uuid,
                    sync_config=sync_cfg_id,
                    question_id=mem.original_question_uuid if mem else uuid4().hex,
                )
                await harness.emit(
                    HarnessEventType.OAUTH_REQUESTED,
                    {"oauth_url": oauth_url, "sync_config_id": sync_cfg_id},
                )
                if mem is not None:
                    token = await mem.recv_oauth_callback(
                        question_id=mem.original_question_uuid,
                        oauth_uuid=oauth_uuid,
                    )
                    if token is not None:
                        existing_creds[sync_cfg_id] = {real_conn: token}

        authorized_rids = set(public_rids)
        for sync_cfg_id, conn_creds in existing_creds.items():
            if sync_cfg_id not in by_conn:
                continue
            real_conn = sync_driver.sync_configs[sync_cfg_id][0]
            token = conn_creds.get(real_conn)
            if not token:
                continue
            conn_rids = by_conn[sync_cfg_id]
            valid = await sync_driver.validate_resources(
                conn_rids,
                credentials=token,
                connection_id=real_conn,
                sync_config_id=sync_cfg_id,
                sync_metadata_by_resource={
                    rid: sync_meta[rid] for rid in conn_rids if rid in sync_meta
                },
            )
            authorized_rids.update(valid)

        return list(
            itertools.chain(
                (rid for rid in candidates if rid in authorized_rids),
            )
        )

    def build_skills(self) -> Sequence[HarnessSkill]:
        base = list(super().build_skills())
        base.append(
            HarnessSkill(
                id="nucliadb_sync_guide",
                name="NucliaDB Sync ACL & OAuth Guide",
                description="Handling per-user OAuth authorization on synced NucliaDB resources.",
                instructions=(
                    "Connector `{namespace}` enforces per-user OAuth ACL checks on synced "
                    "external documents. If zero authorized resources match after authentication, "
                    "inform the user that no accessible documents matched their permissions."
                ),
                recommended_tools=("ask", "find"),
            )
        )
        return base


__all__ = [
    "AdvancedAskInput",
    "AskByTitleInput",
    "AskInput",
    "CatalogInput",
    "CatalogOutput",
    "FindInput",
    "FindOutput",
    "ImagesByTitleInput",
    "LabelsInput",
    "LabelsOutput",
    "NucliaDBSyncToolkit",
    "NucliaDBSyncToolkitConfig",
    "NucliaDBToolkit",
    "NucliaDBToolkitConfig",
    "TitleSearchInput",
]
