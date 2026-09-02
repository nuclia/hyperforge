import os
from copy import deepcopy

import pytest
from hyperforge.engine import main as arag_main
from hyperforge.minimal_fixtures import cassette_nua_key

NUA_KEY = os.environ.get(
    "NUA_KEY",
) or cassette_nua_key("https://europe-1.dp.progress.cloud/")

CONFIG = {
    "drivers": [],
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
    "preprocess": [],
    "context": [
        {
            "module": "context_conditional",
            "title": "",
            "prompt": "The question is about Atlas",
            "then": [
                {
                    "module": "static",
                    "title": "Static Agent",
                    "context": "Atlas is Carmen's dog. He is small and very cute but a bit moody. He was a rescue dog.",
                }
            ],
            "else_": [
                {
                    "module": "static",
                    "title": "Static Agent",
                    "context": "RAO is a product of Progress agentic RAG, it is a powerful agent orchestrator that you are using right now.",
                }
            ],
        },
    ],
    "generation": [
        {"module": "summarize"},
    ],
    "postprocess": [],
}


@pytest.mark.asyncio
@pytest.mark.vcr(ignore_localhost=True)
async def test_conditional_static():
    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="Who is Atlas?",
        config=CONFIG,
        user_metadata={"fullname": "Carmen", "age": "99"},
        loaded_modules=[
            "hyperforge_conditional",
            "hyperforge_static",
            "hyperforge_summarize",
        ],
    )

    keywords = ["Atlas", "Carmen", "dog"]

    assert question_memory.final_answer
    assert all(
        keyword.lower() in question_memory.final_answer.lower() for keyword in keywords
    )

    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="What is RAO?",
        config=CONFIG,
        user_metadata={"fullname": "Carmen", "age": "99"},
        loaded_modules=[
            "hyperforge_conditional",
            "hyperforge_static",
            "hyperforge_summarize",
        ],
    )

    keywords = ["RAO", "powerful", "agent", "orchestrator"]

    assert question_memory.final_answer
    assert all(
        keyword.lower() in question_memory.final_answer.lower() for keyword in keywords
    )


AGENTS_CONTEXT_CONDITIONAL = {
    "context": [
        {
            "module": "static",
            "title": "Static Agent",
            "context": "The best cardamom buns are in Denmark, the most famous place is a bakery called Juno.",
            "next_agent": {
                "module": "context_conditional",
                "title": "",
                "prompt": "The context mentions Juno",
                "on": "CONTEXT",
                "then": [
                    {
                        "module": "static",
                        "title": "Static Agent",
                        "context": "Juno is a bakery in Denmark famous for its cardamom buns. They are the best in the world. The also have a wide variety of other delicious pastries. And great coffee.",
                    }
                ],
                "else_": [
                    {
                        "module": "static",
                        "title": "Static Agent",
                        "context": "You can find good cardamom buns in many places. Some people say the best ones are in Fabrique in New York, but that is subjective. The best way to find good cardamom buns is to try them in different bakeries and see which one you like the most.",
                    }
                ],
            },
        },
    ],
    "generation": [
        {"module": "summarize"},
    ],
}


@pytest.mark.asyncio
@pytest.mark.vcr(ignore_localhost=True)
async def test_conditional_context():
    config = deepcopy(CONFIG)
    config["context"] = AGENTS_CONTEXT_CONDITIONAL["context"]
    config["generation"] = AGENTS_CONTEXT_CONDITIONAL["generation"]
    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="Where can I find good cardamom buns? give me as much info as possible",
        config=config,
        user_metadata={"fullname": "Carmen", "age": "99"},
        loaded_modules=[
            "hyperforge_conditional",
            "hyperforge_static",
            "hyperforge_summarize",
        ],
    )

    assert (
        question_memory.final_answer and "juno" in question_memory.final_answer.lower()
    )
    config["context"][0]["context"] = (
        "Cardamom buns are a type of sweet roll that is flavored with cardamom. They are popular in many countries, including Sweden, Denmark, and Finland. They are often enjoyed with coffee or tea."
    )

    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="Where can I find good cardamom buns?",
        config=config,
        user_metadata={"fullname": "Carmen", "age": "99"},
        loaded_modules=[
            "hyperforge_conditional",
            "hyperforge_static",
            "hyperforge_summarize",
        ],
    )

    assert question_memory.final_answer

    assert "juno" not in question_memory.final_answer.lower()


AGENTS_FALLBACK = {
    "context": [
        {
            "module": "context_conditional",
            "title": "",
            "prompt": "The question is about Atlas",
            "then": [
                {
                    "module": "static",
                    "title": "Static Agent",
                    "context": "Atlas is Carmen's dog. He is small and very cute but a bit moody. He was a rescue dog.",
                }
            ],
            "else_": [
                {
                    "module": "static",
                    "title": "Static Agent",
                    "context": "RAO is a product of Progress agentic RAG, it is a powerful agent orchestrator that you are using right now.",
                }
            ],
            "fallback": {
                "module": "static",
                "title": "Fallback Agent",
                "context": "You are probably asking about where to find the best cookies in the world. The best cookies are the ones made at home with love, but if you want to buy them, you can find great cookies in many bakeries around the world. Some famous ones are Levain Bakery in New York, Lune Croissanterie in Melbourne, and Maison Pichard in Paris.",
            },
        },
    ],
    "generation": [
        {"module": "summarize"},
    ],
}


@pytest.mark.asyncio
@pytest.mark.vcr(ignore_localhost=True)
async def test_conditional_fallback():
    config = deepcopy(CONFIG)
    config["context"] = AGENTS_FALLBACK["context"]
    config["generation"] = AGENTS_FALLBACK["generation"]
    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="Where to have the best cookies?",
        config=config,
        user_metadata={"fullname": "Carmen", "age": "99"},
        loaded_modules=[
            "hyperforge_conditional",
            "hyperforge_static",
            "hyperforge_summarize",
        ],
    )

    keywords = ["cookies", "Levain Bakery", "Lune Croissanterie", "Maison Pichard"]

    assert question_memory.final_answer
    assert all(
        keyword.lower() in question_memory.final_answer.lower() for keyword in keywords
    )


AGENTS_FALLBACK_2 = {
    "context": [
        {
            "module": "context_conditional",
            "title": "",
            "prompt": "The question is about Atlas",
            "then": [
                {
                    "module": "static",
                    "title": "Static Agent",
                    "context": "Atlas is Carmen's dog. He is small and very cute but a bit moody. He was a rescue dog.",
                }
            ],
            "else_": [
                {
                    "module": "static",
                    "title": "Static Agent",
                    "context": "RAO is a product of Progress agentic RAG, it is a powerful agent orchestrator that you are using right now.",
                    "fallback": {
                        "module": "static",
                        "title": "Fallback Agent",
                        "context": "You are probably asking about where to find the best cookies in the world. The best cookies are the ones made at home with love, but if you want to buy them, you can find great cookies in many bakeries around the world. Some famous ones are Levain Bakery in New York, Lune Croissanterie in Melbourne, and Maison Pichard in Paris.",
                    },
                },
            ],
        },
    ],
    "generation": [
        {"module": "summarize"},
    ],
}


@pytest.mark.asyncio
@pytest.mark.vcr(ignore_localhost=True)
async def test_conditional_fallback_second_level():
    config = deepcopy(CONFIG)
    config["context"] = AGENTS_FALLBACK["context"]
    config["generation"] = AGENTS_FALLBACK["generation"]
    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="Where to have the best cookies?",
        config=config,
        user_metadata={"fullname": "Carmen", "age": "99"},
        loaded_modules=[
            "hyperforge_conditional",
            "hyperforge_static",
            "hyperforge_summarize",
        ],
    )

    keywords = ["cookies", "Levain Bakery", "Lune Croissanterie", "Maison Pichard"]

    assert question_memory.final_answer
    assert all(
        keyword.lower() in question_memory.final_answer.lower() for keyword in keywords
    )
