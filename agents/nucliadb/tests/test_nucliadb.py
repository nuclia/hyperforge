import os
from copy import deepcopy
from unittest.mock import AsyncMock, Mock

import pytest
from hyperforge.engine import main as arag_main
from hyperforge.interaction import AragAnswer
from hyperforge.models import Context
from hyperforge.utils.http import PrivateUrlError
from hyperforge_nucliadb.advanced_ask_agent import build_ask_request
from hyperforge_nucliadb.advanced_ask_config import AdvancedAskAgentConfig
from hyperforge_nucliadb.ask.hydrate import hydrate_images
from hyperforge_nucliadb.driver_config import (
    NucliaDBConnection,
)
from nucliadb_models import filters as ndb_filters
from nucliadb_models.hydration import (
    Hydrated,
    HydratedFileField,
    HydratedParagraph,
    HydratedParagraphImage,
    HydratedParagraphPage,
    HydratedParagraphTable,
)
from nucliadb_models.search import AskRequest, Image

NUA_KEY = os.environ.get("NUA_KEY", "DUMMY")
KB_DE48CFAA_3209_4041_BB64_8604AFF061FB = os.environ.get(
    "KB_DE48CFAA_3209_4041_BB64_8604AFF061FB", "DUMMY"
)
KB_DF8B4C24_2807_4888_AD6C_AE97357A638B = os.environ.get(
    "KB_DF8B4C24_2807_4888_AD6C_AE97357A638B", "DUMMY"
)

KB_F718BA84_2973_462F_9B15_F300BD260134 = os.environ.get(
    "KB_F718BA84_2973_462F_9B15_F300BD260134", "DUMMY"
)


CONFIG = {
    "drivers": [
        {
            "name": "nuclia-conversation",
            "provider": "nucliadb",
            "identifier": "nuclia-conversation",
            "config": {
                "url": "https://europe-1.stashify.cloud/api",
                "manager": "https://europe-1.stashify.cloud/api",
                "kbid": "de48cfaa-3209-4041-bb64-8604aff061fb",
                "key": KB_DE48CFAA_3209_4041_BB64_8604AFF061FB,
                "filters": [],
                "description": "Make Discourse Conversation",
            },
        },
        {
            "name": "nuclia-docs",
            "provider": "nucliadb",
            "identifier": "nuclia-docs",
            "config": {
                "url": "https://europe-1.nuclia.cloud/api",
                "manager": "https://europe-1.nuclia.cloud/api",
                "kbid": "df8b4c24-2807-4888-ad6c-ae97357a638b",
                "key": KB_DF8B4C24_2807_4888_AD6C_AE97357A638B,
                "filters": [],
                "description": "Documentation of the Nuclia API, recipies, reference",
            },
        },
        {
            "name": "nuclia-web",
            "provider": "nucliadb",
            "identifier": "nuclia-web",
            "config": {
                "url": "https://europe-1.nuclia.cloud/api",
                "manager": "https://europe-1.nuclia.cloud/api",
                "kbid": "f718ba84-2973-462f-9b15-f300bd260134",
                "key": KB_F718BA84_2973_462F_9B15_F300BD260134,
                "filters": [],
                "description": "Blog post and Main Web Page of Nuclia RAG as a Service, articles, use cases and comercial information",
            },
        },
    ],
    "rules": {
        "rules": [
            {"prompt": "Be polite"},
            {
                "prompt": "The documentation of Nuclia is hosted at https://docs.nuclia.dev"
            },
        ]
    },
    "memory": {},
    "workflow": {
        "id": "default",
        "name": "Default workflow",
        "description": "Default workflow for testing",
        "parameters": {},
    },
    "preprocess": [
        {
            "module": "rephrase",
            "kb": "nuclia-docs",
            "rids": [],
            "labels": [],
            "synonyms": False,
            "extend": True,
            "history": True,
            "session_info": True,
        }
    ],
    "context": [
        {"module": "restricted", "code": "chunks.append(question)"},
        {
            "module": "conditional",
            "prompt": "Its a question about API/SD Usage?",
            "then": {
                "module": "ask",
                "title": "",
                "filter": "/classification",
                "sources": ["nuclia-docs"],
            },
        },
    ],
    "generation": [
        {"module": "summarize"},
    ],
    "postprocess": [],
}


CONFIG_SIMPLE = {
    "drivers": [
        {
            "name": "nuclia-docs",
            "provider": "nucliadb",
            "identifier": "nuclia-docs",
            "config": {
                "url": "https://europe-1.nuclia.cloud/api",
                "manager": "https://europe-1.nuclia.cloud/api",
                "kbid": "df8b4c24-2807-4888-ad6c-ae97357a638b",
                "key": KB_DF8B4C24_2807_4888_AD6C_AE97357A638B,
                "filters": [],
                "description": "Documentation of the Nuclia API, recipies, reference",
            },
        },
    ],
    "rules": {
        "rules": [
            {"prompt": "Be polite"},
        ]
    },
    "memory": {},
    "workflow": {
        "id": "default",
        "name": "Default workflow",
        "description": "Default workflow for testing",
        "parameters": {},
    },
    "preprocess": [
        {
            "module": "rephrase",
        }
    ],
    "context": [
        {
            "module": "ask",
            "title": "",
            "sources": ["nuclia-docs"],
        },
    ],
    "generation": [
        {"module": "summarize"},
    ],
    "postprocess": [],
}


CONFIG_LOCAL = {
    "drivers": [
        {
            "name": "nuclia-docs",
            "provider": "nucliadb",
            "identifier": "nuclia-docs",
            "config": {
                "url": "http://localhost:8080/api",
                "manager": "http://localhost:8080/api",
                "kbid": "df8b4c24-2807-4888-ad6c-ae97357a638b",
                "key": KB_DF8B4C24_2807_4888_AD6C_AE97357A638B,
                "filters": [],
                "description": "Documentation of the Nuclia API, recipies, reference",
            },
        }
    ],
    "rules": {
        "rules": [
            {"prompt": "Be polite"},
        ]
    },
    "memory": {},
    "workflow": {
        "id": "default",
        "name": "Default workflow",
        "description": "Default workflow for testing",
        "parameters": {},
    },
    "preprocess": [
        {
            "module": "rephrase",
        }
    ],
    "context": [
        {
            "module": "ask",
            "title": "",
            "sources": ["nuclia-docs"],
        },
    ],
    "generation": [
        {"module": "summarize"},
    ],
    "postprocess": [],
}


@pytest.mark.skip
async def test_nucliadb_agent():
    answers = []

    async def callback(obj: AragAnswer):
        answers.append(obj)

    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="Como usar max_tokens.answer",
        config=CONFIG,
        callback=callback,
        loaded_modules=[
            "hyperforge_nucliadb",
            "hyperforge_rephrase",
            "hyperforge_conditional",
            "hyperforge_summarize",
        ],
    )

    assert question_memory.final_answer and "max_tokens" in question_memory.final_answer


@pytest.mark.asyncio
@pytest.mark.vcr(ignore_localhost=True)
async def test_nucliadb_agent_simple():
    answers = []

    async def callback(obj: AragAnswer):
        answers.append(obj)

    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="Como usar max_tokens.answer. En español y dame link a la doucmentación",
        config=CONFIG_SIMPLE,
        callback=callback,
        loaded_modules=[
            "hyperforge_nucliadb",
            "hyperforge_rephrase",
            "hyperforge_conditional",
            "hyperforge_summarize",
        ],
    )

    assert question_memory.final_answer and "max_tokens" in question_memory.final_answer


@pytest.mark.asyncio
@pytest.mark.vcr(ignore_localhost=True)
async def test_nucliadb_agent_simple_disable_ai_parameter_search():
    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="Como usar max_tokens.answer. En español y dame link a la doucmentación",
        config=CONFIG_SIMPLE,
        loaded_modules=[
            "hyperforge_nucliadb",
            "hyperforge_summarize",
            "hyperforge_rephrase",
        ],
    )
    assert question_memory.final_answer and "max_tokens" in question_memory.final_answer


@pytest.mark.asyncio
@pytest.mark.vcr(ignore_localhost=True)
async def test_nucliadb_agent_basic_ask():
    config = deepcopy(CONFIG_SIMPLE)

    config["context"][0]["module"] = "basic_ask"

    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="Como usar max_tokens.answer. En español y dame link a la doucmentación",
        config=config,
        loaded_modules=[
            "hyperforge_nucliadb",
            "hyperforge_rephrase",
            "hyperforge_conditional",
            "hyperforge_summarize",
        ],
    )

    assert question_memory.final_answer and "max_tokens" in question_memory.final_answer


@pytest.mark.asyncio
async def test_nucliadb_hydrate_images():
    # resource ids
    rid = "aaaaaa0123456789abcdef0123456789"
    rad = "bbbbbb0123456789abcdef0123456789"
    rod = "cccccc0123456789abcdef0123456789"

    chunk_ids = [
        f"{rid}/f/ocr-doc/0-100",
        f"{rad}/f/tables-doc/0-100",
        f"{rod}/f/paged/0-100",
    ]

    context = Context(
        question="a silly question",
        source="my imagination",
        agent="super",
        original_question_uuid=None,
        actual_question_uuid=None,
    )

    nucliadb_driver = AsyncMock()
    resp = Mock()
    nucliadb_driver.driver.session.post.return_value = resp

    hydrated = Hydrated(
        resources={},
        fields={
            f"{rid}/f/ocr-doc": HydratedFileField(
                id=f"{rid}/f/ocr-doc",
                resource=rid,
            ),
            f"{rad}/f/tables-doc": HydratedFileField(
                id=f"{rad}/f/tables-doc",
                resource=rad,
                previews={
                    "16": Image(
                        content_type="image/png", b64encoded="table page preview"
                    ),
                },
            ),
            f"{rod}/f/paged": HydratedFileField(
                id=f"{rod}/f/paged",
                resource=rod,
                previews={
                    "10": Image(content_type="image/png", b64encoded="page preview"),
                },
            ),
        },
        paragraphs={
            f"{rid}/f/ocr-doc/0-100": HydratedParagraph(
                id=f"{rid}/f/ocr-doc/0-100",
                field=f"{rid}/f/ocr-doc",
                resource=rid,
                image=HydratedParagraphImage(
                    source_image=Image(
                        content_type="image/png", b64encoded="source image"
                    )
                ),
            ),
            f"{rad}/f/tables-doc/0-100": HydratedParagraph(
                id=f"{rad}/f/tables-doc/0-100",
                field=f"{rad}/f/tables-doc",
                resource=rad,
                table=HydratedParagraphTable(
                    page_preview_ref="16",
                ),
            ),
            f"{rod}/f/paged/0-100": HydratedParagraph(
                id=f"{rod}/f/paged/0-100",
                field=f"{rod}/f/paged",
                resource=rod,
                page=HydratedParagraphPage(
                    page_preview_ref="10",
                ),
            ),
        },
    )

    resp.status_code = 500
    await hydrate_images(
        chunk_ids,
        context,
        nucliadb_driver,
        vllm=True,
        visual=True,
    )
    assert context.images == {}

    resp.status_code = 200
    resp.json.return_value = hydrated.model_dump()
    await hydrate_images(
        chunk_ids,
        context,
        nucliadb_driver,
        vllm=True,
        visual=True,
    )
    expected = {
        f"{rid}/f/ocr-doc/0-100": Image(
            content_type="image/png", b64encoded="source image"
        ),
        f"{rad}/f/tables-doc/0-100": Image(
            content_type="image/png", b64encoded="table page preview"
        ),
        f"{rod}/f/paged/0-100": Image(
            content_type="image/png", b64encoded="page preview"
        ),
    }
    # we need a deep comparison as Image instances are not comparable
    assert context.images.keys() == expected.keys()
    for key, value in expected.items():
        image = context.images[key]
        assert image.content_type == value.content_type
        assert image.b64encoded == value.b64encoded


def test_build_ask_request():
    agent = AdvancedAskAgentConfig(
        filter_expression=ndb_filters.FilterExpression(
            field=ndb_filters.FieldMimetype(type="application", subtype="pdf")
        ),
        search_configuration="foobar",
    )
    driver = NucliaDBConnection(
        url="https://europe-1.nuclia.cloud/api",
        manager="foo",
        kbid="df8b4c24-2807-4888-ad6c-ae97357a638b",
        description="foo",
        filters=["/l/foo/bar", "/e/CITY/NewYork", "/n/i/application/pdf"],
    )
    request: AskRequest = build_ask_request(
        agent,
        driver,
        question="What is Nuclia?",
    )
    assert request.query == "What is Nuclia?"
    assert request.filters == []
    assert request.filter_expression is not None
    assert isinstance(request.filter_expression.field, ndb_filters.And)
    assert len(request.filter_expression.field.operands) == 2
    assert request.filter_expression.field.operands[0] == ndb_filters.FieldMimetype(
        type="application", subtype="pdf"
    )
    assert request.filter_expression.field.operands[1].operands[0] == ndb_filters.Label(
        labelset="foo", label="bar"
    )
    assert request.filter_expression.field.operands[1].operands[
        1
    ] == ndb_filters.Entity(subtype="CITY", value="NewYork")
    assert request.filter_expression.field.operands[1].operands[
        2
    ] == ndb_filters.FieldMimetype(type="application", subtype="pdf")
    assert request.search_configuration == "foobar"


@pytest.mark.asyncio
async def test_nucliadb_not_localhost():
    with pytest.raises(PrivateUrlError):
        await arag_main(
            agent_id="default",
            internal_nua=False,
            external_nua_api_key=NUA_KEY,
            question="Como usar max_tokens.answer. En español y dame link a la doucmentación",
            config=CONFIG_LOCAL,
            loaded_modules=[
                "hyperforge_nucliadb",
                "hyperforge_summarize",
                "hyperforge_rephrase",
            ],
        )
