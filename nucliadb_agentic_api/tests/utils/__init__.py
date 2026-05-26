import uuid

from nucliadb_protos import resources_pb2 as rpb
from nucliadb_protos.writer_pb2 import BrokerMessage, OpStatusWriter
from nucliadb_protos.writer_pb2_grpc import WriterStub

from nucliadb_agentic_api.tests.utils.broker_messages import BrokerMessageBuilder
from nucliadb_agentic_api.tests.utils.dirty_index import mark_dirty


def broker_resource(
    kbid: str,
    rid: str | None = None,
    slug: str | None = None,
    source: BrokerMessage.MessageSource.ValueType = BrokerMessage.MessageSource.WRITER,
) -> BrokerMessage:
    """
    Returns a broker resource with barebones metadata.
    """
    rid = rid or str(uuid.uuid4()).replace("-", "")
    slug = slug or f"{rid}slug1"

    bmb = BrokerMessageBuilder(kbid=kbid, rid=rid, slug=slug, source=source)
    bmb.with_title("Title Resource")
    bmb.with_summary("Summary of document")
    bm = bmb.build()
    return bm


def broker_resource_with_title_paragraph(
    kbid: str,
    rid: str | None = None,
    slug: str | None = None,
) -> BrokerMessage:
    """
    Returns a broker resource with barebones metadata.
    """
    rid = rid or str(uuid.uuid4()).replace("-", "")
    slug = slug or f"{rid}slug1"

    bmb = BrokerMessageBuilder(kbid=kbid, rid=rid, slug=slug)

    title_builder = bmb.with_title("Title Resource")
    title_builder.with_extracted_paragraph_metadata(rpb.Paragraph(start=0, end=5))

    bmb.with_summary("Summary of document")

    bm = bmb.build()
    return bm


async def inject_message(
    writer: WriterStub,
    message: BrokerMessage,
    timeout: float | None = None,
    wait_for_ready: bool | None = None,
):
    await mark_dirty()
    resp = await writer.ProcessMessage(
        iter([message]), timeout=timeout, wait_for_ready=wait_for_ready
    )  # type: ignore
    assert resp.status == OpStatusWriter.Status.OK
