import asyncio
import itertools
from typing import Dict, List, Optional, cast
from uuid import uuid4

from hyperforge.configure import agent
from hyperforge.interaction import (
    Feedback,
    OAuthAuthenticateURL,
    OAuthFeedbackReturnSchema,
)
from hyperforge.manager import Manager
from hyperforge.memory import QuestionMemory
from hyperforge.models import Context, Source
from hyperforge.settings import OAuthSettings
from nucliadb_models import SyncMetadata
from nucliadb_models.filters import (
    And,
    CatalogFilterExpression,
    FilterExpression,
    Or,
    OriginSource,
)
from nucliadb_models.resource import Resource as NucliaDBResource
from nucliadb_models.search import (
    ResourceProperties,
)

from hyperforge_nucliadb.basic_ask_agent import (
    BasicAskAgent,
    get_ndb_driver,
)
from hyperforge_nucliadb.sync.config import SyncAskAgentConfig
from hyperforge_nucliadb.sync.driver import SyncDriver


@agent(
    id="sync",
    agent_type="context",
    title="Sync Service Ask Agent",
    description="Provide answer questions from synched resources. This agent is responsible for handling the authentication flow and resource validation for synched resources, and then providing the valid resources to the Basic Ask Agent to answer questions.",
    config_schema=SyncAskAgentConfig,
)
class SyncAskAgent(BasicAskAgent):  # type: ignore
    sources: Dict[str, SyncDriver]
    settings: OAuthSettings = OAuthSettings()

    def __init__(
        self, config: SyncAskAgentConfig, agent_id: Optional[str] = None
    ) -> None:
        super().__init__(config, agent_id)
        self.sources = {}

    def get_connections(self, manager: Manager) -> Dict[str, SyncDriver]:
        for source in self.config.sources:
            driver = manager.drivers.get(source)
            driver = cast(SyncDriver, driver)
            self.sources[source] = driver
        return self.sources

    async def get_oauth_url(
        self,
        manager: Manager,
        connection_ids: List[str],
        rao_redirect_url: str,
        oauth_uuid: str,
        kb_source_id: str,
        question_id: str,
    ) -> Dict[str, str]:
        urls = {}
        driver = cast(SyncDriver, manager.drivers.get(kb_source_id))
        for connection_id in connection_ids:
            if (
                connection_id in driver.sync_configs
                and len(driver.sync_configs[connection_id]) > 0
            ):
                real_connection_id = driver.sync_configs[connection_id][0]
                url = await driver.get_oauth_url(
                    connection_id=real_connection_id,
                    rao_redirect_url=rao_redirect_url,
                    oauth_uuid=oauth_uuid,
                    sync_config=connection_id,
                    question_id=question_id,
                )
                urls[connection_id] = url
        return urls

    async def validate_resources_by_connection(
        self,
        credentials: str,
        resource_ids: List[str],
        connection: SyncDriver,
        connection_id: str,
        sync_config_id: str,
        sync_metadata_by_resource: Dict[str, SyncMetadata],
    ) -> List[str]:
        # Validate resources
        validated_resources = await connection.validate_resources(
            resource_ids,
            credentials=credentials,
            connection_id=connection_id,
            sync_config_id=sync_config_id,
            sync_metadata_by_resource=sync_metadata_by_resource,
        )
        return validated_resources

    async def post_filter_resources_by_connection(
        self,
        memory: QuestionMemory,
        manager: Manager,
        resources: Dict[str, List[str]],
    ) -> Dict[str, Dict[str, List[str]]]:
        """We have a list of ARAG resources per KB source. Now we need to filter them by connection"""
        filtered_resources: Dict[str, Dict[str, List[str]]] = {}
        connections_by_resource: Dict[str, List[str]] = {}
        sync_metadata_by_resource: Dict[str, SyncMetadata] = {}
        connections = self.get_connections(manager)
        for kb_source_id, resource_ids in resources.items():
            # Get resource source connection for each resource
            ndb = get_ndb_driver(manager, kb_source_id)
            get_resource_ids_tasks = []
            for resource in resource_ids:
                get_resource_ids_tasks.append(
                    ndb.driver.get_resource_by_id(
                        kbid=ndb.config.kbid,
                        rid=resource,
                        query_params={"show": [ResourceProperties.ORIGIN.value]},
                    )
                )
            resource_objs: List[NucliaDBResource] = await asyncio.gather(
                *get_resource_ids_tasks
            )
            for resource_obj in resource_objs:
                if (
                    resource_obj.origin is not None
                    and resource_obj.origin.source_id is not None
                ):
                    source_id = resource_obj.origin.source_id.replace(
                        "sync_config_", ""
                    )
                    connections_by_resource.setdefault(source_id, []).append(
                        resource_obj.id
                    )
                    if resource_obj.origin.sync_metadata is not None:
                        sync_metadata_by_resource[resource_obj.id] = (
                            resource_obj.origin.sync_metadata
                        )

            # Get providers needed for the resources allocated
            needed_providers_ids = list(connections_by_resource.keys())
            driver = self.sources[kb_source_id]

            creds_providers = {}
            for sync_config_id in needed_providers_ids:
                if (
                    sync_config_id not in driver.sync_configs
                    or len(driver.sync_configs[sync_config_id]) == 0
                ):
                    raise Exception(
                        f"Connection ID {sync_config_id} not found in driver sync configs"
                    )
                inner_connection_id = driver.sync_configs[sync_config_id][0]
                creds_providers[sync_config_id] = driver.information[
                    inner_connection_id
                ].provider

            # First message, ask for OAuth credentials
            oauth_credentials_feedback = Feedback(
                question="Get credentials",
                data=None,
                get_credentials=creds_providers,
                module="oauth",
                request_id=memory.get_session_id(),
                agent_id=self.agent_id,
                response_schema=OAuthFeedbackReturnSchema.model_json_schema(),
            )
            answer = await memory.send_feedback(oauth_credentials_feedback)
            if answer is None or answer.request_id is None:
                raise Exception("No answer received for OAuth initiation")
            assert answer.request_id == memory.get_session_id()
            # Validate answer schema
            try:
                oauth_feedback_return = OAuthFeedbackReturnSchema.model_validate_json(
                    answer.response
                )
            except Exception as e:
                raise Exception(f"Invalid OAuth feedback response: {e}")

            existing_credentials: Dict[str, Dict[str, str]] = {}
            if (
                oauth_feedback_return.existing_credentials is None
                or len(oauth_feedback_return.existing_credentials) == 0
            ):
                # The frontend does not have credentials yet, we need to perform the OAuth flow

                oauth_uuid = uuid4().hex
                rao_redirect_url = self.settings.rao_redirect_url.format(
                    agent_id=memory.get_agent_id(),
                    session_id=memory.get_session_id(),
                    oauth_uuid=oauth_uuid,
                    workflow_id=memory.get_workflow_id(),
                )
                # Get all the authorize URLs from all connections on this KB
                urls = await self.get_oauth_url(
                    manager,
                    needed_providers_ids,
                    rao_redirect_url=rao_redirect_url,
                    oauth_uuid=oauth_uuid,
                    kb_source_id=kb_source_id,
                    question_id=memory.original_question_uuid,
                )
                for connection_id, url in urls.items():
                    # For each connection send the OAuth URL to the frontend
                    oauth_authenticate_url = OAuthAuthenticateURL(oauth_url=url)
                    await memory.send_oauth(oauth=oauth_authenticate_url)

                    credential = await memory.recv_oauth_callback(
                        question_id=memory.original_question_uuid, oauth_uuid=oauth_uuid
                    )
                    if credential is not None:
                        inner_conn_id = driver.sync_configs[connection_id][0]
                        existing_credentials[connection_id] = {
                            inner_conn_id: credential
                        }

                # Send OAuth initiation feedback
                oauth_credentials_feedback = Feedback(
                    question="Send credentials",
                    data=None,
                    get_credentials=None,
                    credentials=existing_credentials,
                    module="oauth",
                    request_id=memory.get_session_id(),
                    agent_id=self.agent_id,
                    response_schema=OAuthFeedbackReturnSchema.model_json_schema(),
                )
                answer = await memory.send_feedback(oauth_credentials_feedback)
                if answer is None or answer.request_id is None:
                    raise Exception("No answer received for OAuth initiation")
                assert answer.request_id == memory.get_session_id()
            elif oauth_feedback_return.existing_credentials is not None:
                existing_credentials = oauth_feedback_return.existing_credentials
            else:
                raise Exception("No credentials provided")

            for (
                connection_id,
                connection_credentials,
            ) in existing_credentials.items():
                source = connections[kb_source_id]
                if (
                    connection_id not in source.sync_configs
                    or len(source.sync_configs[connection_id]) == 0
                ):
                    raise Exception("Connection / source not found")
                inner_connection_id = source.sync_configs[connection_id][0]
                if inner_connection_id is None or connection_credentials is None:
                    raise Exception("Connection not found")

                credential = connection_credentials[inner_connection_id]
                filtered_resource_ids = await self.validate_resources_by_connection(
                    connection=source,
                    credentials=credential,
                    resource_ids=resource_ids,
                    connection_id=inner_connection_id,
                    sync_config_id=connection_id,
                    sync_metadata_by_resource=sync_metadata_by_resource,
                )

                filtered_resources.setdefault(kb_source_id, {})[connection_id] = (
                    filtered_resource_ids
                )
        return filtered_resources

    def enrich_filter(self) -> FilterExpression:
        return FilterExpression()

    def enrich_catalog_filter(
        self,
        catalog_filter: Optional[CatalogFilterExpression] = None,
    ):
        if catalog_filter is None:
            catalog_filter = CatalogFilterExpression(
                resource=Or(
                    operands=[
                        OriginSource(id=f"sync_config_{connection_id}")
                        for source in self.config.sources
                        for connection_id in self.sources[source].sync_configs.keys()
                    ]
                )
            )
        else:
            catalog_filter.resource = And(
                operands=[
                    catalog_filter.resource,
                    Or(
                        operands=[
                            OriginSource(id=f"sync_config_{connection_id}")
                            for source in self.config.sources
                            for connection_id in self.sources[
                                source
                            ].sync_configs.keys()
                        ]
                    ),
                ]
            )
        return catalog_filter

    # PUBLIC METHODS USED BY THE BASIC ASK AGENT FRAMEWORK
    async def search_by_title(
        self,
        memory: QuestionMemory,
        manager: Manager,
        title: str,
        filters: Optional[List[str]] = None,
        catalog_filter: Optional[CatalogFilterExpression] = None,
        **kwargs,
    ) -> Dict[str, List[str]]:
        catalog_filter = self.enrich_catalog_filter(catalog_filter)
        resources = await super().search_by_title(
            memory=memory,
            manager=manager,
            title=title,
            filters=filters,
            catalog_filter=catalog_filter,
            **kwargs,
        )
        resources_by_connection = await self.post_filter_resources_by_connection(
            memory=memory,
            manager=manager,
            resources=resources,
        )
        final_resources: Dict[str, List[str]] = {}
        for source_id, connections in resources_by_connection.items():
            for _, resource_ids in connections.items():
                final_resources.setdefault(source_id, []).extend(resource_ids)
        return final_resources

    async def inner_ask_by_title(
        self,
        source_id: str,
        manager: Manager,
        memory: QuestionMemory,
        question: str,
        resource_ids: List[str],
    ) -> Context:
        if len(resource_ids) == 0:
            resource_ids = await self.retrieve(manager, source_id, question)

            resources_by_connection = await self.post_filter_resources_by_connection(
                memory=memory,
                manager=manager,
                resources={source_id: resource_ids},
            )
            resources_by_connection_for_source = resources_by_connection.get(
                source_id, {}
            )
            resource_ids = list(
                itertools.chain(*resources_by_connection_for_source.values())
            )
            if resource_ids is None or len(resource_ids) == 0:
                raise Exception("No valid resources found after OAuth validation")

        return await super().inner_ask_by_title(
            source_id=source_id,
            manager=manager,
            memory=memory,
            question=question,
            resource_ids=resource_ids,
        )

    async def inner_rag(
        self,
        source_obj: Source,
        manager: Manager,
        memory: QuestionMemory,
        question: str,
        question_uuid: Optional[str] = None,
        keyword_filters: List[str] = [],
        and_filters: Optional[List[str]] = None,
        or_filters: Optional[List[str]] = None,
        full_resource: bool = False,
        resource_filters: Optional[List[str]] = None,
    ) -> Context:
        # Retrieve resources as usual, but then filter them by connection and validate them with the SyncDriver
        catalog_filter = self.enrich_filter()
        resources = await self.retrieve(
            manager=manager,
            source_id=source_obj.id,
            question=question,
            keyword_filters=keyword_filters,
            and_filters=and_filters,
            or_filters=or_filters,
            catalog_filter=catalog_filter,
        )
        resources_by_connection = await self.post_filter_resources_by_connection(
            memory=memory,
            manager=manager,
            resources={source_obj.id: resources},
        )
        resources_by_connection_for_source = resources_by_connection.get(
            source_obj.id, {}
        )
        resource_ids = list(
            itertools.chain(*resources_by_connection_for_source.values())
        )
        if resource_ids is None or len(resource_ids) == 0:
            raise Exception("No valid resources found after OAuth validation")

        context = await super().inner_rag(
            source_obj=source_obj,
            manager=manager,
            memory=memory,
            question=question,
            question_uuid=question_uuid,
            keyword_filters=keyword_filters,
            and_filters=and_filters,
            or_filters=or_filters,
            full_resource=full_resource,
            resource_filters=resource_ids,
        )
        return context

    async def rag(
        self,
        source_obj: Source,
        question_uuid: str,
        question: str,
        memory: QuestionMemory,
        manager: Manager,
        flow_id: str,
    ) -> tuple[str, str] | None:
        context = await self.inner_rag(
            source_obj=source_obj,
            manager=manager,
            memory=memory,
            question=question,
            question_uuid=question_uuid,
        )
        missing = await self.save_ctx_and_return_missing(
            context=context,
            question=question,
            memory=memory,
            manager=manager,
            flow_id=flow_id,
        )
        return missing

    async def inner_facets_search(
        self,
        memory: QuestionMemory,
        manager: Manager,
        question: str,
        source: Source,
        idx: int,
    ) -> Context:
        raise NotImplementedError("Facets search not implemented for SyncAskAgent")

    async def inner_facets(
        self,
        memory: QuestionMemory,
        manager: Manager,
        question: str,
        source: Source,
        idx: int,
    ) -> Context:
        raise NotImplementedError("Facets search not implemented for SyncAskAgent")

    async def inner_catalog_search(
        self,
        memory: QuestionMemory,
        manager: Manager,
        question: str,
        source: Source,
        idx: int,
    ) -> Context:
        raise NotImplementedError("Facets search not implemented for SyncAskAgent")
