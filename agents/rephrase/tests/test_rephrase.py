import os
from copy import deepcopy

import pytest
from hyperforge.engine import main as arag_main
from hyperforge.minimal_fixtures import cassette_nua_key

NUA_KEY = os.environ.get(
    "NUA_KEY",
) or cassette_nua_key("https://europe-1.nuclia.cloud/")


DE48CFAA_3209_4041_BB64_8604AFF061FB = os.environ.get(
    "KB_DE48CFAA_3209_4041_BB64_8604AFF061FB"
) or cassette_nua_key("https://europe-1.nuclia.cloud/")

DF8B4C24_2807_4888_AD6C_AE97357A638B = os.environ.get(
    "KB_DF8B4C24_2807_4888_AD6C_AE97357A638B"
) or cassette_nua_key("https://europe-1.nuclia.cloud/")
pytestmark = [
    pytest.mark.vcr(
        ignore_localhost=True, ignore_hosts=["test", "europe-1.nuclia.cloud"]
    ),
    pytest.mark.asyncio,
]


CONFIG = {
    "drivers": [
        {
            "name": "nuclia-conversation",
            "provider": "nucliadb",
            "identifier": "nucliadb-1",
            "config": {
                "url": "https://europe-1.stashify.cloud/api",
                "manager": "https://europe-1.stashify.cloud/api",
                "kbid": "de48cfaa-3209-4041-bb64-8604aff061fb",
                "key": DE48CFAA_3209_4041_BB64_8604AFF061FB,
                "filters": [],
                "description": "Make Discourse Conversation",
            },
        },
        {
            "name": "nuclia-docs",
            "provider": "nucliadb",
            "identifier": "nucliadb-2",
            "config": {
                "identifier": "nucliadb-2",
                "url": "https://europe-1.nuclia.cloud/api",
                "manager": "https://europe-1.nuclia.cloud/api",
                "kbid": "df8b4c24-2807-4888-ad6c-ae97357a638b",
                "key": DF8B4C24_2807_4888_AD6C_AE97357A638B,
                "filters": [],
                "description": "Documentation of the Nuclia API, recipies, reference",
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
            "kb": "nucliadb-2",
            "rids": [],
            "labels": [],
            "synonyms": False,
            "extend": True,
            "history": False,  # Requires NucliaDB to be running to retrieve memory
            "session_info": True,
        }
    ],
    "context": [
        {
            "module": "ask",
            "title": "",
            "sources": ["nucliadb-2"],
            "ai_parameter_search": False,
        },
    ],
    "generation": [
        {"module": "summarize"},
    ],
    "postprocess": [],
}


async def test_rephrase_agent():
    config = deepcopy(CONFIG)
    config["preprocess"][0]["split_question"] = True
    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="Como usar max_tokens.answer. En español y dame link a la doucmentación",
        config=config,
        loaded_modules=[
            "hyperforge_rephrase",
            "hyperforge_nucliadb",
            "hyperforge_summarize",
        ],
    )
    assert (
        len(question_memory.context_questions) <= 5
    )  # Rephrase should not make more than this number of questions
    assert question_memory.final_answer and "max_tokens" in question_memory.final_answer


async def test_rephrase_agent_only_rephrase():
    config = deepcopy(CONFIG)
    config["preprocess"][0]["split_question"] = False
    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="Como usar max_tokens.answer. En español y dame link a la doucmentación",
        config=config,
        loaded_modules=[
            "hyperforge_rephrase",
            "hyperforge_nucliadb",
            "hyperforge_summarize",
        ],
    )

    assert (
        len(question_memory.context_questions) == 1
    )  # Rephrase should not make more than this number of questions
