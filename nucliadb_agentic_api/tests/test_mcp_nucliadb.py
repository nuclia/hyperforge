import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hyperforge.configure import get_driver_config_instance
from hyperforge.llm import NUAConnection
from hyperforge.manager import Manager
from hyperforge.memory.memory import EphemeralSessionMemory, MemoryConfig
from hyperforge.models import Rule, Rules
from hyperforge_mcp.agent import MCPAgent
from hyperforge_mcp.config import MCPAgentConfig, Transport
from mcp.server.fastmcp.exceptions import ResourceError
from mcp.types import TextContent
from nuclia import REGIONAL
from nucliadb_agentic_api.ask.predict import (
    start_predict_engine,
    stop_predict_engine,
)
from nucliadb_sdk.v2.exceptions import NotFoundError, RateLimitError
from nucliadb_utils.settings import nuclia_settings

NUA_KEY = os.environ.get("NUA_KEY", "DUMMY")

pytestmark = [
    pytest.mark.vcr(
        ignore_localhost=True,  # Ignore localhost requests (e.g., to the test server)
    ),
    pytest.mark.asyncio,
]

DRIVERS = [
    {
        "provider": "mcphttp",
        "identifier": "mcphttp-01",
        "name": "mcphttp",
        "config": {
            "uri": "http://localhost:3001",
            "headers": {
                "X-STF-USER": "user@example.com",
                "X-STF-ROLES": "READER",
                "X-STF-ACCOUNT": "account_id",
                "X-STF-ACCOUNT-TYPE": "personal",
            },
        },
    },
]


MEMORY = {"nucliadb": {"url": "", "key": "", "kbid": ""}}

ROUTER = {
    "key": NUA_KEY,
}


RULES: Rules = Rules(
    rules=[
        Rule(prompt="Be polite"),
    ]
)


async def test_mcp_nucliadb_generation_client(
    arag_api_http: str,
    article_dataset: str,
    disable_safe_transport,
):
    DRIVERS[0]["config"]["uri"] = (  # type: ignore
        f"http://{arag_api_http}/api/v1/kb/{article_dataset}/mcp"
    )

    nua_driver = await NUAConnection.model_validate(ROUTER).connect()

    if "http" in nua_driver.region:
        url = nua_driver.region.strip("/")
    else:
        url = REGIONAL.format(region=nua_driver.region).strip("/")

    nuclia_settings.onprem = True
    nuclia_settings.nuclia_public_url = url
    nuclia_settings.nuclia_service_account = ROUTER["key"]
    nuclia_settings.nuclia_zone = nua_driver.region
    await start_predict_engine()
    manager = await Manager.from_config(
        drivers=[get_driver_config_instance(driver) for driver in DRIVERS],
        nua=nua_driver,
    )
    mcp_client = await MCPAgent.from_config(
        MCPAgentConfig(
            title="MCP NucliaDB Agent",
            transport=Transport.HTTP,
            source="mcphttp-01",
            context_validation_model="claude-4-5-haiku",
        )
    )
    memory = EphemeralSessionMemory.from_config(
        MemoryConfig.model_validate(MEMORY),
        agent_id="agent",
        rules=RULES,
        workflow_id="default",
    )
    memory.init("hola")
    question = "what is the architecture of Agents?"
    question_memory = memory.start_question(question, question_id="question_id")

    await mcp_client.initialize(manager, question_memory)

    await mcp_client.get_question_context(
        memory=question_memory,
        manager=manager,
        question_uuid="question_id",
        question=question,
        flow_id="flow_id",
    )

    assert len(question_memory.contexts) > 0, "No context retrieved from MCP NucliaDB"
    assert question_memory.contexts[0].chunks[0].metadata["type"] == "citations", (  # type: ignore
        "Retrieved context is not of type 'answer'"
    )
    assert question_memory.contexts[0].chunks[-1].metadata["type"] == "answer", (  # type: ignore
        "Retrieved context is not of type 'answer'"
    )
    assert question_memory.contexts[0].summary, "Answer summary is empty"

    question = (
        "What is the architecture of Agents with the filter //metadata/type=citations? "
    )
    question_memory = memory.start_question(question, question_id="question_id")
    await mcp_client.get_question_context(
        memory=question_memory,
        manager=manager,
        question_uuid="question_id",
        question=question,
        flow_id="flow_id",
    )
    assert len(question_memory.contexts) == 0

    await stop_predict_engine()


async def test_mcp_nucliadb_get_document(
    arag_api_http: str, article_dataset: str, disable_safe_transport
):
    DRIVERS[0]["config"]["uri"] = (  # type: ignore
        f"http://{arag_api_http}/api/v1/kb/{article_dataset}/mcp"
    )

    nua_driver = await NUAConnection.model_validate(ROUTER).connect()

    if "http" in nua_driver.region:
        url = nua_driver.region.strip("/")
    else:
        url = REGIONAL.format(region=nua_driver.region).strip("/")

    nuclia_settings.onprem = True
    nuclia_settings.nuclia_public_url = url
    nuclia_settings.nuclia_service_account = ROUTER["key"]
    nuclia_settings.nuclia_zone = nua_driver.region
    await start_predict_engine()
    manager = await Manager.from_config(
        drivers=[get_driver_config_instance(driver) for driver in DRIVERS],
        nua=nua_driver,
    )
    mcp_client = await MCPAgent.from_config(
        MCPAgentConfig(
            title="MCP NucliaDB Agent",
            transport=Transport.HTTP,
            source="mcphttp-01",
            context_validation_model="claude-4-5-haiku",
            prune_context=False,
        )
    )
    memory = EphemeralSessionMemory.from_config(
        MemoryConfig.model_validate(MEMORY),
        agent_id="agent",
        rules=RULES,
        workflow_id="default",
    )
    memory.init("hola")
    question = "full content of the document 72f6d5fe65e5441f97e52e19216460c3"
    question_memory = memory.start_question(question, question_id="question_id")

    await mcp_client.initialize(manager, question_memory)
    await mcp_client.get_question_context(
        memory=question_memory,
        manager=manager,
        question_uuid="question_id",
        question=question,
        flow_id="flow_id",
    )

    assert "Used tool: get_document" in question_memory.steps[1].value, (
        "get_document tool was not used in the second step"
    )
    assert len(question_memory.contexts[0].chunks) > 0, (
        "No chunks retrieved from MCP NucliaDB"
    )
    # Check that at least one of the chunks contains the expected text content from the documents
    expected_text = "Agentic Context Engineering"
    assert any(
        expected_text in chunk.text for chunk in question_memory.contexts[0].chunks
    ), "Expected text content from documents not found in retrieved chunks"

    await stop_predict_engine()


async def test_mcp_nucliadb_batch_get_documents(
    arag_api_http: str, article_dataset: str, disable_safe_transport
):
    DRIVERS[0]["config"]["uri"] = (  # type: ignore
        f"http://{arag_api_http}/api/v1/kb/{article_dataset}/mcp"
    )

    nua_driver = await NUAConnection.model_validate(ROUTER).connect()

    if "http" in nua_driver.region:
        url = nua_driver.region.strip("/")
    else:
        url = REGIONAL.format(region=nua_driver.region).strip("/")

    nuclia_settings.onprem = True
    nuclia_settings.nuclia_public_url = url
    nuclia_settings.nuclia_service_account = ROUTER["key"]
    nuclia_settings.nuclia_zone = nua_driver.region
    await start_predict_engine()
    manager = await Manager.from_config(
        drivers=[get_driver_config_instance(driver) for driver in DRIVERS],
        nua=nua_driver,
    )
    mcp_client = await MCPAgent.from_config(
        MCPAgentConfig(
            title="MCP NucliaDB Agent",
            transport=Transport.HTTP,
            source="mcphttp-01",
            context_validation_model="claude-4-5-haiku",
            prune_context=False,
        )
    )
    memory = EphemeralSessionMemory.from_config(
        MemoryConfig.model_validate(MEMORY),
        agent_id="agent",
        rules=RULES,
        workflow_id="default",
    )
    memory.init("hola")
    question = "full content of the documents 72f6d5fe65e5441f97e52e19216460c3 and c0a5192da40540a8a86c9c1b0114aaee"
    question_memory = memory.start_question(question, question_id="question_id")

    await mcp_client.initialize(manager, question_memory)
    await mcp_client.get_question_context(
        memory=question_memory,
        manager=manager,
        question_uuid="question_id",
        question=question,
        flow_id="flow_id",
    )
    assert "Used tool: batch_get_documents" in question_memory.steps[1].value, (
        "batch_get_documents tool was not used in the second step"
    )
    assert len(question_memory.contexts[0].chunks) > 1, (
        "Not enough chunks retrieved from MCP NucliaDB"
    )
    # Check that at least one of the chunks contains the expected text content from the documents
    expected_texts = [
        "Agentic Context Engineering",
        "From Shallow Loops to Deep Agents",
    ]
    assert any(
        any(expected_text in chunk.text for expected_text in expected_texts)
        for chunk in question_memory.contexts[0].chunks
    ), "Expected text content from documents not found in retrieved chunks"
    await stop_predict_engine()


async def test_mcp_nucliadb_two_steps(
    arag_api_http: str, article_dataset: str, disable_safe_transport
):
    DRIVERS[0]["config"]["uri"] = (  # type: ignore
        f"http://{arag_api_http}/api/v1/kb/{article_dataset}/mcp"
    )

    nua_driver = await NUAConnection.model_validate(ROUTER).connect()

    if "http" in nua_driver.region:
        url = nua_driver.region.strip("/")
    else:
        url = REGIONAL.format(region=nua_driver.region).strip("/")

    nuclia_settings.onprem = True
    nuclia_settings.nuclia_public_url = url
    nuclia_settings.nuclia_service_account = ROUTER["key"]
    nuclia_settings.nuclia_zone = nua_driver.region
    await start_predict_engine()
    manager = await Manager.from_config(
        drivers=[get_driver_config_instance(driver) for driver in DRIVERS],
        nua=nua_driver,
    )
    mcp_client = await MCPAgent.from_config(
        MCPAgentConfig(
            title="MCP NucliaDB Agent",
            transport=Transport.HTTP,
            source="mcphttp-01",
            context_validation_model="claude-4-5-sonnet",
            prune_context=False,
            tool_choice_model="claude-4-5-sonnet",
        )
    )
    memory = EphemeralSessionMemory.from_config(
        MemoryConfig.model_validate(MEMORY),
        agent_id="agent",
        rules=RULES,
        workflow_id="default",
    )
    memory.init("hola")
    question = "Full document that talks about the architecture of Agents. Retrieve the whole thing"
    question_memory = memory.start_question(question, question_id="question_id")

    await mcp_client.initialize(manager, question_memory)
    await mcp_client.get_question_context(
        memory=question_memory,
        manager=manager,
        question_uuid="question_id",
        question=question,
        flow_id="flow_id",
    )
    necessary_steps = [
        "Used tool: search_documents with arguments",
        "Used tool: get_document",
    ]
    all_steps_values = " ".join(step.value for step in question_memory.steps)
    assert all(
        necessary_step in all_steps_values for necessary_step in necessary_steps
    ), "Not all necessary steps were used in the question memory"
    assert len(question_memory.contexts[0].chunks) > 0, (
        "No chunks retrieved from MCP NucliaDB"
    )
    # Check that at least one of two of the chunks contains the expected text content from the documents
    expected_texts = [
        "Research Quantum Computing",
        "architecture of Agents 2.0",
    ]
    assert any(
        any(expected_text in chunk.text for expected_text in expected_texts)
        for chunk in question_memory.contexts[0].chunks
    ), "Expected text content from documents not found in retrieved chunks"


async def test_text_content_audience_includes_assistant():
    from hyperforge.api.v1.mcp_nucliadb import call_tool, get_resource

    # --- test get_resource ---
    field = MagicMock()
    field.extracted.text.text = "some content"
    resource = MagicMock()
    resource.data.generics = {"f1": field}
    resource.data.texts = None
    resource.data.files = None
    resource.data.links = None
    resource.data.conversations = None

    ndb = AsyncMock()
    ndb.get_resource_by_id.return_value = resource

    results = await get_resource(ndb, "kbid", "rid")
    assert all(isinstance(r, TextContent) for r in results)
    for r in results:
        assert "assistant" in r.annotations.audience, (
            "get_resource: 'assistant' missing from TextContent audience"
        )

    # --- test call_tool search_documents answer block ---
    resp = MagicMock()
    resp.answer = "the answer"
    resp.citation_footnote_to_context = {"block-AB": "rid123/t/body/0-10"}
    resp.retrieval_results.resources = {"rid123": MagicMock()}
    resp.augmented_context = MagicMock()
    resp.augmented_context.fields = {}
    paragraph = MagicMock()
    paragraph.text = "citation text"
    resp.augmented_context.paragraphs = {"rid123/t/body/0-10": paragraph}

    resource = MagicMock()
    resource.computedmetadata = None
    resource.title = "title"
    resp.retrieval_results.resources["rid123"] = resource

    ask_result_mock = MagicMock()
    ask_result_mock.model_dump = AsyncMock(return_value=resp)

    ask_mock = AsyncMock(return_value=ask_result_mock)

    with patch(
        "nuclia_arag_api.v1.mcp_nucliadb.ask",
        new=ask_mock,
    ):
        results = await call_tool(
            x_stf_account="account",
            x_nucliadb_user="user",
            x_ndb_client=MagicMock(),
            x_forwarded_for="",
            ndb_reader=AsyncMock(),
            ndb_search=AsyncMock(),
            kbid="kbid",
            name="search_documents",
            arguments={"query": "test", "search_configuration": "my-config"},
        )

    # Verify search_configuration was correctly injected
    ask_args = ask_mock.call_args[1]
    assert ask_args["ask_request"].search_configuration == "my-config"

    answer_blocks = [r for r in results if isinstance(r, TextContent)]
    assert answer_blocks, "No TextContent returned from call_tool search_documents"
    for r in answer_blocks:
        assert "assistant" in r.annotations.audience, (
            "call_tool search_documents: 'assistant' missing from TextContent audience"
        )

    answer_block = answer_blocks[-1]
    assert "Document IDs:" in answer_block.text
    assert "block-AB: rid123" in answer_block.text
    assert answer_block.meta == {
        "type": "answer",
        "document_ids": {"block-AB": "rid123"},
    }

    # --- test call_tool search_documents without search_configuration ---
    ask_mock_no_config = AsyncMock(return_value=ask_result_mock)
    with patch(
        "nuclia_arag_api.v1.mcp_nucliadb.ask",
        new=ask_mock_no_config,
    ):
        await call_tool(
            x_stf_account="account",
            x_nucliadb_user="user",
            x_ndb_client=MagicMock(),
            x_forwarded_for="",
            ndb_reader=AsyncMock(),
            ndb_search=AsyncMock(),
            kbid="kbid",
            name="search_documents",
            arguments={"query": "test"},
        )
    # Validate search_configuration was NOT passed to ask_request (defaults to None or whatever AskRequest default is)
    ask_args_no_config = ask_mock_no_config.call_args[1]
    assert ask_args_no_config["ask_request"].search_configuration is None

    # --- test call_tool search_documents with invalid configuration ---
    # An unrecognised exception from `ask` is wrapped into a ResourceError.
    ask_mock_invalid = AsyncMock(side_effect=Exception("Invalid configuration"))
    with (
        patch("nuclia_arag_api.v1.mcp_nucliadb.ask", new=ask_mock_invalid),
        pytest.raises(
            ResourceError,
            match="Search failed: Invalid configuration",
        ),
    ):
        await call_tool(
            x_stf_account="account",
            x_nucliadb_user="user",
            x_ndb_client=MagicMock(),
            x_forwarded_for="",
            ndb_reader=AsyncMock(),
            ndb_search=AsyncMock(),
            kbid="kbid",
            name="search_documents",
            arguments={"query": "test", "search_configuration": "non-existent-config"},
        )


async def test_call_tool_search_documents_known_exceptions():
    """Known ask() exceptions are converted to descriptive ResourceErrors."""
    from hyperforge.api.v1.mcp_nucliadb import call_tool

    from nucliadb_agentic_api.src.nucliadb_agentic_api.ask.exceptions import (
        InvalidQueryError,
        KnowledgeBoxNotFound,
        NoRetrievalResultsError,
    )

    base_args = dict(
        x_stf_account="account",
        x_nucliadb_user="user",
        x_ndb_client=MagicMock(),
        x_forwarded_for="",
        ndb_reader=AsyncMock(),
        ndb_search=AsyncMock(),
        kbid="my-kbid",
        name="search_documents",
        arguments={"query": "test"},
    )

    cases = [
        (KnowledgeBoxNotFound(), "my-kbid"),
        (InvalidQueryError("filters", "bad value"), "Invalid search query"),
        (NoRetrievalResultsError(), "No documents matched"),
        (RateLimitError("limit"), "Service limit reached"),
        (NotFoundError("not found"), "Resource not found during search"),
    ]

    for exc, expected_fragment in cases:
        with (
            patch(
                "nuclia_arag_api.v1.mcp_nucliadb.ask", new=AsyncMock(side_effect=exc)
            ),
            pytest.raises(ResourceError, match=expected_fragment),
        ):
            await call_tool(**base_args)


async def test_call_tool_invalid_arguments():
    """Pydantic validation errors are wrapped into ResourceError with a descriptive message."""
    from hyperforge.api.v1.mcp_nucliadb import call_tool

    with pytest.raises(ResourceError, match="Invalid arguments for 'search_documents'"):
        await call_tool(
            x_stf_account="account",
            x_nucliadb_user="user",
            x_ndb_client=MagicMock(),
            x_forwarded_for="",
            ndb_reader=AsyncMock(),
            ndb_search=AsyncMock(),
            kbid="kbid",
            name="search_documents",
            arguments={"query": ""},
        )


async def test_call_tool_batch_get_documents_partial_failure():
    """A single failing document does not abort the whole batch; the error is reported inline."""
    from hyperforge.api.v1.mcp_nucliadb import call_tool

    good_field = MagicMock()
    good_field.extracted.text.text = "good content"
    good_resource = MagicMock()
    good_resource.data.generics = {"f1": good_field}
    good_resource.data.texts = None
    good_resource.data.files = None
    good_resource.data.links = None
    good_resource.data.conversations = None

    ndb_reader = AsyncMock()

    def get_resource_by_id_side_effect(rid, kbid, query_params):
        if rid == "good-rid":
            return good_resource
        raise NotFoundError("not found")

    ndb_reader.get_resource_by_id.side_effect = get_resource_by_id_side_effect

    results = await call_tool(
        x_stf_account="account",
        x_nucliadb_user="user",
        x_ndb_client=MagicMock(),
        x_forwarded_for="",
        ndb_reader=ndb_reader,
        ndb_search=AsyncMock(),
        kbid="kbid",
        name="batch_get_documents",
        arguments={"names": ["good-rid", "bad-rid"]},
    )

    texts = [r.text for r in results if isinstance(r, TextContent)]
    assert any("good content" in t for t in texts), "Good document content missing"
    assert any("Error fetching" in t and "bad-rid" in t for t in texts), (
        "Expected inline error message for the failed document"
    )


async def test_get_resource_not_found():
    """NotFoundError from NucliaDB becomes a descriptive ResourceError."""
    from hyperforge.api.v1.mcp_nucliadb import get_resource

    ndb = AsyncMock()
    ndb.get_resource_by_id.side_effect = NotFoundError("not found")

    with pytest.raises(ResourceError, match="Document not found.*missing-rid"):
        await get_resource(ndb, "kbid", "missing-rid")


async def test_call_tool_unknown_tool():
    """Requesting an unknown tool raises ResourceError, not a bare Exception."""
    from hyperforge.api.v1.mcp_nucliadb import call_tool

    with pytest.raises(ResourceError, match="Unknown tool"):
        await call_tool(
            x_stf_account="account",
            x_nucliadb_user="user",
            x_ndb_client=MagicMock(),
            x_forwarded_for="",
            ndb_reader=AsyncMock(),
            ndb_search=AsyncMock(),
            kbid="kbid",
            name="nonexistent_tool",
            arguments={},
        )


async def test_read_resource_malformed_uri():
    """A URI without the right structure raises a descriptive ResourceError."""
    from hyperforge.api.v1.mcp_nucliadb import read_resource

    ndb = AsyncMock()

    with pytest.raises(ResourceError, match="Malformed resource URI"):
        await read_resource(ndb, "kbid", "nucliadb://no-slash-here")

    with pytest.raises(ResourceError, match="Unknown resource URI scheme"):
        await read_resource(ndb, "kbid", "http://example.com/resource")


async def test_mcp_protected_resource_metadata(
    arag_api_http_client,
    article_dataset: str,
):
    kbid = article_dataset
    resp = await arag_api_http_client.get(
        f"/.well-known/oauth-protected-resource/api/v1/kb/{kbid}/mcp"
    )
    assert resp.status_code == 200
    body = resp.json()
    expected_url = f"{arag_api_http_client.base_url}/api/v1/kb/{kbid}/mcp"
    expected_url = expected_url.replace("http://", "https://")
    assert body == {
        "resource": expected_url,
        "scopes_supported": ["offline_access", "openid"],
        "authorization_servers": ["https://oauth.progress.cloud"],
    }
