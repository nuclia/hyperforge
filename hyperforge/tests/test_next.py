import os

import pytest
from hyperforge.engine import main as arag_main
from hyperforge.interaction import AragAnswer

NUA_KEY = os.environ.get("NUA_KEY", "DUMMY")

KB_DF8B4C24_2807_4888_AD6C_AE97357A638B = os.environ.get(
    "KB_DF8B4C24_2807_4888_AD6C_AE97357A638B", "DUMMY"
)

CONFIG = {
    "drivers": [
        {
            "name": "nuclia-docs",
            "provider": "nucliadb",
            "identifier": "nuclia-docs",
            "config": {
                "url": "https://europe-1.rag.progress.cloud/api",
                "manager": "https://europe-1.rag.progress.cloud/api",
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
        }
    ],
    "context": [
        {
            "module": "basic_ask",
            "title": "",
            "sources": ["nuclia-docs"],
            "next_agent": {
                "module": "static",
                "title": "",
                "context": "Cardamom bun is a pastry",
            },
        },
    ],
    "generation": [
        {"module": "summarize"},
    ],
    "postprocess": [],
}


@pytest.mark.asyncio
@pytest.mark.vcr(ignore_localhost=True)
async def test_nucliadb_next():
    answers = []

    async def callback(obj: AragAnswer):
        answers.append(obj)

    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="How to use max_tokens in Nuclia? and what is a Cardamom bun",
        config=CONFIG,
        callback=callback,
        loaded_modules=[
            "hyperforge_static",
            "hyperforge_nucliadb",
            "hyperforge_rephrase",
            "hyperforge_summarize",
        ],
    )
    keywords = ["max_tokens", "Nuclia", "cardamom", "pastry"]
    assert question_memory.final_answer
    assert all(
        keyword.lower() in question_memory.final_answer.lower() for keyword in keywords
    )
