import asyncio
from asyncio import gather
from functools import reduce
from typing import Dict, List, Optional

import httpx
from hyperforge.configure import driver
from hyperforge.driver import Driver
from hyperforge.models import Facets
from hyperforge.utils.http import SafeTransport
from nuclia.lib.nua_responses import StoredLearningConfiguration
from nucliadb_models.graph.requests import (
    GraphNodesSearchRequest,
    GraphRelationsSearchRequest,
    GraphSearchRequest,
)
from nucliadb_models.graph.responses import (
    GraphNodesSearchResponse,
    GraphRelationsSearchResponse,
    GraphSearchResponse,
)
from nucliadb_models.resource import Resource
from nucliadb_models.search import (
    AskRequest,
    CatalogFacetsPrefix,
    CatalogFacetsRequest,
    CatalogFacetsResponse,
    CatalogRequest,
    CatalogResponse,
    FindOptions,
    FindRequest,
    KnowledgeboxFindResults,
    MinScore,
    SearchOptions,
    SearchRequest,
    SyncAskResponse,
)
from nucliadb_sdk.v2 import NucliaDBAsync

from hyperforge import logger
from hyperforge_nucliadb.driver_config import (
    ManagerConnection,
    NucliaDBConfig,
    NucliaDBConnection,
)

DEFAULT_FIELD_FACETS = [
    "/n/icon",
    "/n/metadata.status",
    "/metadata.language",
    "/metadata.languages",
    "/field",
    "/f",
    "/ml",
    "/origin.tags",
    "/origin.metadata",
    "/origin.path",
    "/origin.source-id",
]
DEFAULT_CHUNK_LABELS = ["/k"]


async def connect(conn: NucliaDBConnection):
    headers: Dict[str, str] = {}
    if "http://localhost" in conn.url:
        headers = {
            "X-NUCLIADB-ROLES": "READER",
        }
    return NucliaDBAsync(
        api_key=conn.key,
        url=conn.url,
        headers=headers,
        _httpx_transport=SafeTransport(),
    )


async def manager_connect(conn: NucliaDBConnection):
    headers: Dict[str, str] = {}
    return ManagerConnection(api_key=conn.key, url=conn.manager, headers=headers)


@driver(
    id="nucliadb",
    title="NucliaDB Driver",
    description="Driver for interacting with the NucliaDB API.",
    config_schema=NucliaDBConfig,
)
class NucliaDBDriver(Driver):
    driver: NucliaDBAsync
    manager: ManagerConnection
    config: NucliaDBConnection
    _synonyms: Optional[dict[str, list[str]]]

    @classmethod
    async def init(cls, driver: NucliaDBConfig):
        return cls(
            provider=driver.provider,
            name=driver.name,
            config=driver.config,
            driver=await connect(driver.config),
            manager=await manager_connect(driver.config),
            _synonyms=None,
        )

    async def synonyms_raw(self) -> Dict[str, List[str]]:
        synonyms_list = await self.driver.get_custom_synonyms(kbid=self.config.kbid)
        return dict(
            (k.lower(), [x.lower() for x in v])
            for k, v in synonyms_list.synonyms.items()
        )

    async def labels(self) -> Dict[str, List[str]]:
        labelsets = await self.driver.get_labelsets(kbid=self.config.kbid)
        result = {}
        for labelset in labelsets.labelsets:
            labels = await self.driver.get_labelset(
                kbid=self.config.kbid, labelset=labelset
            )
            result[labelset] = [x.title for x in labels.labels]

        return result

    async def field_facets(self) -> Dict[str, int]:
        field_labels = {}
        search_fulltext_classify = await self.driver.search(
            kbid=self.config.kbid,
            content=SearchRequest(
                faceted=["/classification.labels"],
                features=[SearchOptions.FULLTEXT],
            ),
        )

        if (
            search_fulltext_classify.fulltext is not None
            and search_fulltext_classify.fulltext.facets is not None
            and "/classification.labels" in search_fulltext_classify.fulltext.facets
        ):
            fulltext_labelsets = [
                x
                for x in search_fulltext_classify.fulltext.facets[
                    "/classification.labels"
                ].keys()
            ]
        else:
            fulltext_labelsets = []

        paths = [path for path in DEFAULT_FIELD_FACETS]

        paths.extend(fulltext_labelsets)

        search_results_second = await self.driver.search(
            kbid=self.config.kbid,
            content=SearchRequest(faceted=paths, features=[SearchOptions.FULLTEXT]),
        )
        if (
            search_results_second.fulltext is not None
            and search_results_second.fulltext.facets is not None
        ):
            field_labels = search_results_second.fulltext.facets

        field_labels = reduce(lambda a, b: {**a, **b}, field_labels.values())
        return field_labels

    async def field_labels(
        self, labelsets: list[str] | None = None
    ) -> tuple[dict[str, dict[str, int]], int]:
        """
        Returns a prettified and filtered list of the field labels facets.

        Args:
            labelsets (list[str] | None): List of labelsets to filter the results.

        Returns:
            dict[str, dict[str, int]]: Dictionary of labelsets with their labels and counts
            int: Total number of resources for the knowledge box.
        """
        # Add the filter to the facets
        prefixes = (
            [f"/classification.labels/{ls}" for ls in labelsets]
            if labelsets
            else ["/classification.labels"]
        )
        catalog_result = await self.driver.catalog(
            kbid=self.config.kbid,
            faceted=prefixes,
            page_size=0,
        )
        if not catalog_result.fulltext or not catalog_result.fulltext.facets:
            raise Exception("Empty facets in catalog result")

        # Remove /classification.labels/ prefix from the facets
        result: dict[str, dict[str, int]] = {
            labelset.split("/")[-1]: {
                label.split(f"{labelset}/")[-1]: count
                for label, count in facets.items()
            }
            for labelset, facets in catalog_result.fulltext.facets.items()
        }
        return result, catalog_result.fulltext.total

    async def paragraph_facets(self) -> Dict[str, int]:
        paragraphs_facets = {}

        search_paragraph_classify = await self.driver.search(
            kbid=self.config.kbid,
            content=SearchRequest(
                faceted=["/classification.labels"],
                features=[SearchOptions.KEYWORD],
            ),
        )

        if (
            search_paragraph_classify.paragraphs is not None
            and search_paragraph_classify.paragraphs.facets is not None
            and "/classification.labels" in search_paragraph_classify.paragraphs.facets
        ):
            paragraph_labels = [
                x
                for x in search_paragraph_classify.paragraphs.facets[
                    "/classification.labels"
                ].keys()
            ]
        else:
            paragraph_labels = []

        paths = [path for path in DEFAULT_CHUNK_LABELS]

        if "/classification.labels" in paths:
            paths.remove("/classification.labels")

        paths.extend(paragraph_labels)

        search_results_second = await self.driver.search(
            kbid=self.config.kbid, content=SearchRequest(faceted=paths)
        )
        if (
            search_results_second.paragraphs is not None
            and search_results_second.paragraphs.facets is not None
        ):
            paragraphs_facets = search_results_second.paragraphs.facets

        try:
            paragraphs_facets = reduce(
                lambda a, b: {**a, **b}, paragraphs_facets.values()
            )
        except Exception as e:
            logger.error(f"Error reducing paragraph facets: {e}")
            paragraphs_facets = {}

        return paragraphs_facets

    async def facets(self) -> Facets:
        chunks, fields = await gather(self.paragraph_facets(), self.field_facets())
        return Facets(chunks=chunks, fields=fields)

    async def facets_native(self) -> CatalogFacetsResponse:
        """
        Get facets for the knowledge box.
        This is a native implementation that returns the facets as a dictionary.

        We restrict the types of facets to a predefined subset, as on big knowledge boxes, the response could be very large.
        """
        facets = CatalogFacetsResponse(facets={})

        # Run requests in parallel as otherwise the facets endpoint can take too long on large knowledge boxes.
        tasks = []
        for prefix in [
            "/n/i",  # content types
            "/s/p",  # primary language
            # "/l",  # classification labels
        ]:
            request = CatalogFacetsRequest(
                prefixes=[CatalogFacetsPrefix(prefix=prefix)]
            )
            tasks.append(
                self.driver.catalog_facets(kbid=self.config.kbid, content=request)
            )

        responses = await asyncio.gather(*tasks, return_exceptions=True)
        for facets_response in responses:
            if isinstance(facets_response, CatalogFacetsResponse):
                facets.facets.update(facets_response.facets)
            else:
                logger.error(f"Error getting facets: {facets_response}")
        return facets

    async def description(self):
        kb_obj = await self.driver.get_knowledge_box(kbid=self.config.kbid)
        return f"""{self.config.description} {kb_obj.config.description if kb_obj.config else ""}"""

    async def get_learning_configuration(self):
        configuration = await self.driver.get_configuration(kbid=self.config.kbid)
        if configuration is None:
            raise Exception("No configuration found for the knowledge box")

        return StoredLearningConfiguration.model_validate(configuration)

    async def synonyms(self, sentence: str) -> str:
        synonyms_obj = await self.driver.get_custom_synonyms(kbid=self.config.kbid)
        self._synonyms = synonyms_obj.model_dump()
        words = sentence.split()
        result = []
        if self._synonyms is not None:
            for word in words:
                if word in self._synonyms:
                    word = " ".join(self._synonyms[word])
                result.append(word)
        return " ".join(result)

    async def ask(self, item: AskRequest, headers={}) -> SyncAskResponse:
        return await self.driver.ask(
            kbid=self.config.kbid, content=item, headers=headers
        )

    async def find(
        self, q: str, filters: List[str] = [], rids: List[str] = []
    ) -> KnowledgeboxFindResults:
        return await self.driver.find(
            kbid=self.config.kbid,
            content=FindRequest(
                query=q,
                resource_filters=rids,
                features=[FindOptions.KEYWORD],
                min_score=MinScore(bm25=1.0),
                filters=list(set(self.config.filters + filters)),
            ),
        )

    async def find_raw(self, q: FindRequest) -> KnowledgeboxFindResults:
        return await self.driver.find(
            kbid=self.config.kbid,
            content=q,
        )

    async def graph_relations(
        self, q: GraphRelationsSearchRequest
    ) -> GraphRelationsSearchResponse:
        return await self.driver.graph_relations(kbid=self.config.kbid, content=q)

    async def graph_search(self, q: GraphSearchRequest) -> GraphSearchResponse:
        return await self.driver.graph_search(kbid=self.config.kbid, content=q)

    async def graph_nodes(self, q: GraphNodesSearchRequest) -> GraphNodesSearchResponse:
        return await self.driver.graph_nodes(kbid=self.config.kbid, content=q)

    async def catalog_search_raw(self, q: CatalogRequest) -> CatalogResponse:
        return await self.driver.catalog(content=q, kbid=self.config.kbid)

    async def get_resource_by_id(
        self, rid: str, query_params: Optional[Dict[str, str]] = None
    ) -> Optional[Resource]:
        return await self.driver.get_resource_by_id(
            kbid=self.config.kbid, rid=rid, query_params=query_params
        )

    async def get_ephemeral_token(self, path: Optional[str] = None) -> Optional[str]:
        """Get an ephemeral token scoped to the knowledge box, optionally restricted to a specific resource path.

        Args:
            path: When provided, the token will only be valid for this specific path
                  (e.g. /kb/{kbid}/resource/{rid}). This prevents the token from
                  being used to access the entire knowledge box.
        """
        async with httpx.AsyncClient(transport=SafeTransport()) as client:
            headers = {
                "Authorization": f"Bearer {self.config.key}",
                "Content-Type": "application/json",
                "Accept": "*/*",
            }
            url = f"{self.manager.url}/v1/ephemeral_token"
            body: dict = {"ttl": 600}
            if path is not None:
                body["path"] = path
            response = await client.post(url, headers=headers, json=body)
            if response.status_code != 201:
                raise Exception(f"Error getting ephemeral token: {response.text}")
            data = response.json()
            return data.get("token", None)


def format_ndb_labels(
    labels: Dict[str, List[str]],
    max_examples: int = 5,
) -> str:
    """Format labels and their examples into a string representation."""
    labels_str = ""
    for label, examples in labels.items():
        labels_str += f"- {label}: {', '.join(examples[:max_examples]) + (', ...' if len(examples) > max_examples else '')}\n"
    return labels_str


def format_ndb_catalog(
    catalog: CatalogResponse,
) -> str:
    """Format catalog results into a string representation."""
    catalog_str = ""
    # TODO: Consider converting to jinja2 template
    for rid, resource in catalog.resources.items():
        catalog_str += f"- Title: {resource.title}\n"
        if resource.summary:
            catalog_str += f"  Summary: '{resource.summary[:100]}'\n"
        catalog_str += f"  Format: {resource.icon}\n"
        if resource.metadata and resource.metadata.language:
            catalog_str += f"  Language: {resource.metadata.language}\n"

        if resource.usermetadata and resource.usermetadata.classifications:
            catalog_str += "  User defined Labels: "
            catalog_str += " | ".join(
                f'{classification.labelset}:"{classification.label}"'
                for classification in resource.usermetadata.classifications
                # XXX: Don't know if this is needed
                if not classification.cancelled_by_user
            )
            catalog_str += "\n"

        if (
            resource.computedmetadata
            and resource.computedmetadata.field_classifications
        ):
            catalog_str += "  Computed Labels: "
            catalog_str += " | ".join(
                f'{classification.labelset}:"{classification.label}"'
                for field_classification in resource.computedmetadata.field_classifications
                for classification in field_classification.classifications
            )
            catalog_str += "\n"
        catalog_str += "\n"
    return catalog_str
