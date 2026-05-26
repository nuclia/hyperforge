import pytest
from httpx import AsyncClient
from nucliadb_protos.resources_pb2 import FieldType
from nucliadb_protos.utils_pb2 import Relation, RelationMetadata, RelationNode
from nucliadb_protos.writer_pb2 import BrokerMessage, FieldComputedMetadataWrapper

from nucliadb_agentic_api.tests.utils import inject_message
from nucliadb_agentic_api.tests.utils.dirty_index import wait_for_sync


@pytest.fixture(scope="function")
async def graph_resource(
    nucliadb_writer: AsyncClient, nucliadb_ingest_grpc, knowledgebox: str
):
    kbid = knowledgebox
    resp = await nucliadb_writer.post(
        f"/kb/{kbid}/resources",
        json={
            "title": "Knowledge graph",
            "slug": "knowledgegraph",
            "summary": "Test knowledge graph",
            "texts": {
                "inception1": {
                    "body": "Christopher Nolan directed Inception. Very interesting movie."
                },
                "inception2": {"body": "Leonardo DiCaprio starred in Inception."},
                "inception3": {"body": "Joseph Gordon-Levitt starred in Inception."},
                "leo": {
                    "body": "Leonardo DiCaprio is a great actor. DiCaprio started acting in 1989."
                },
            },
        },
    )
    assert resp.status_code == 201
    rid = resp.json()["uuid"]

    nodes = {
        "nolan": RelationNode(
            value="Christopher Nolan",
            ntype=RelationNode.NodeType.ENTITY,
            subtype="DIRECTOR",
        ),
        "inception": RelationNode(
            value="Inception", ntype=RelationNode.NodeType.ENTITY, subtype="MOVIE"
        ),
        "leo": RelationNode(
            value="Leonardo DiCaprio",
            ntype=RelationNode.NodeType.ENTITY,
            subtype="ACTOR",
        ),
        "dicaprio": RelationNode(
            value="DiCaprio", ntype=RelationNode.NodeType.ENTITY, subtype="ACTOR"
        ),
        "levitt": RelationNode(
            value="Joseph Gordon-Levitt",
            ntype=RelationNode.NodeType.ENTITY,
            subtype="ACTOR",
        ),
    }

    edges = [
        Relation(
            relation=Relation.RelationType.ENTITY,
            source=nodes["nolan"],
            to=nodes["inception"],
            relation_label="directed",
            metadata=RelationMetadata(
                # Set this field id as int enum value since this is how legacy relations reported paragraph_id
                paragraph_id=rid + "/4/inception1/0-37",
                data_augmentation_task_id="my_graph_task_id",
            ),
        ),
        Relation(
            relation=Relation.RelationType.ENTITY,
            source=nodes["leo"],
            to=nodes["inception"],
            relation_label="starred",
            metadata=RelationMetadata(
                paragraph_id=rid + "/t/inception2/0-39",
                data_augmentation_task_id="my_graph_task_id",
            ),
        ),
        Relation(
            relation=Relation.RelationType.ENTITY,
            source=nodes["levitt"],
            to=nodes["inception"],
            relation_label="starred",
            metadata=RelationMetadata(
                paragraph_id=rid + "/t/inception3/0-42",
            ),
        ),
        Relation(
            relation=Relation.RelationType.ENTITY,
            source=nodes["leo"],
            to=nodes["dicaprio"],
            relation_label="analogy",
            metadata=RelationMetadata(
                paragraph_id=rid + "/t/leo/0-70",
                data_augmentation_task_id="my_graph_task_id",
            ),
        ),
    ]

    # Add relations to the resource as processor does
    for n in nodes.values():
        edges.append(
            Relation(
                relation=Relation.RelationType.ENTITY,
                source=RelationNode(value=rid, ntype=RelationNode.NodeType.RESOURCE),
                to=n,
            )
        )

    bm = BrokerMessage()
    bm.uuid = rid
    bm.kbid = kbid
    bm.source = BrokerMessage.MessageSource.PROCESSOR
    fcmw = FieldComputedMetadataWrapper()
    fcmw.field.field_type = FieldType.TEXT
    fcmw.field.field = "inception1"
    fcmw.metadata.metadata.relations.add(relations=edges)
    bm.field_metadata.append(fcmw)
    await inject_message(nucliadb_ingest_grpc, bm)
    await wait_for_sync()
    yield rid
