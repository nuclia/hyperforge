import os

import pytest
from hyperforge.engine import main as arag_main
from hyperforge.interaction import AragAnswer

NUA_KEY = os.environ.get("NUA_KEY", "DUMMY")

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
                "key": "eyJhbGciOiJSUzI1NiIsImtpZCI6InNhIiwidHlwIjoiSldUIn0.eyJpc3MiOiJodHRwczovL2V1cm9wZS0xLm51Y2xpYS5jbG91ZC8iLCJpYXQiOjE3NTkyNTQ3NjQsInN1YiI6ImUwNGUwMzcyLTYwNDgtNDY5ZC04NWExLWI1ZTM1MjBmMzdlZiIsImp0aSI6IjgzMzRhY2NlLTIwMTUtNGY0MS05M2U5LTczNzk1MTg2NDdiZiIsImV4cCI6MTc5MDc5MDc2MCwia2V5IjoiMmYxMzYyNTItNjNiMy00NzA1LTg1MjQtMDhmYWJjOWUzMjUyIiwia2lkIjoiN2RmMzY2NDctOTdiOC00NzU0LWExNjUtZWZkY2ZlMDRkMzI2In0.kwHAfx9RRTI-G3S64X0iisr0iAyXRKNRhnN4C67MkLSxeu1AOAnVV8EIQuu4jpXW7O4FkSsthFXEv9ZxlRRh_CaS0z_TjPzIzDPeE6eIKskZ70Q7c-pDe949WE9DZiDyy9_dwKsdX5cnvYpKorp0ROm-GvRXrdHaTZKDSYWht3gvEtm6-0j9C1gx2BzKr2coizUAIde_qjSpLOojO4S-k8P8I9dsQFagdcrjxgGWgrAzjhAs_qkqlRmP0QP6S7ToN0nrbHmtKKb0lWmcpVvlAfH95CM20YUs7IAqU_t7-_V6mm43FstRgGeiHkoapo8nPVJtXMBSlaM7GSz0Kxf2TWQwi94mTEQLdA8CblX0skMCfIHFwbcbm1Vf-2C6LywAsSmTYAwsVPpqeQcVZdrfLMhddCjZKUFCNLSurCSb4TuN79GZicPCJDT-VEBMlNH8ayHOyRib5RyqvgXUwGN9zyM-ma7RrVk4eEwSk7923bn_9GTk-s5tYw_exbYsQ1Qa84GA6NzgJ_kNQmgJwb2zW1V5ddCpYd5k6lNEdPRk0JQKlCC2zTmSvnRcLxfDPi4SZFdLLdtG0j2hIl_QNTEC_3VtqJds4FMofy7TkmUObdbEmXjdAsOxkqj2ntGOsaBNiCI_w47BbPvG_V1LsBHDrrIo0Wo1fgAhUbtWV7Dd5J4",
                "filters": [],
                "description": "Documentation of the Nuclia API, recipies, reference",
            },
        },
        {
            "name": "perplexity",
            "provider": "perplexity",
            "identifier": "perplexity",
            "config": {"key": "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"},
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
                "module": "perplexity",
                "title": "Perplexity Agent",
                "domain": [],
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
    )
    keywords = ["max_tokens", "Nuclia", "cardamom", "pastry"]
    assert question_memory.final_answer
    assert all(
        keyword.lower() in question_memory.final_answer.lower() for keyword in keywords
    )
