import os
from copy import deepcopy

import pytest

from hyperforge.engine import main as arag_main
from hyperforge.minimal_fixtures import cassette_nua_key
from hyperforge.models import HistoryQuestionAnswer

NUA_KEY = os.environ.get("NUA_KEY") or cassette_nua_key(
    "https://europe-1.nuclia.cloud/"
)

pytestmark = [
    pytest.mark.vcr(ignore_localhost=True),
    pytest.mark.asyncio,
]

# Static context text injected as the retrieval result.
STATIC_CONTEXT = (
    "The max_tokens parameter controls how many tokens the LLM may generate "
    "in a single response. It can be set per-request via the API or configured "
    "globally in the agent workflow. The default value depends on the model, but "
    "is typically 1024 tokens."
)

CONFIG = {
    "drivers": [],
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
            "rids": [],
            "labels": [],
            "synonyms": False,
            "extend": False,
            "history": True,  # uses context_history() — picks up client chat_history
            "session_info": False,
            "split_question": False,
        }
    ],
    "context": [
        {
            "module": "static",
            "title": "max_tokens reference",
            "context": STATIC_CONTEXT,
            "prune_context": False,
        }
    ],
    "generation": [
        {
            "module": "summarize",
            "conversational": True,  # uses get_chat_history() — picks up client chat_history
        }
    ],
    "postprocess": [],
}


async def test_chat_history_is_used_in_rephrase_and_summarize():
    """Client-provided chat_history overrides empty server-side session history.

    The previous exchange is about ``max_tokens``; the follow-up question uses
    a pronoun ("it") that only makes sense in that context.  The rephrase agent
    receives the history via ``context_history()`` and should expand the
    question; the summarize agent receives it via ``get_chat_history()``.
    """
    prior_history = [
        HistoryQuestionAnswer(
            question="What is the max_tokens parameter in Nuclia?",
            answer="max_tokens controls the maximum number of tokens the LLM may generate in a single response.",
        ),
        HistoryQuestionAnswer(
            question="Oh nice, can you always start you responses with a smiley emoji?",
            answer="Sure! 😊 I can start my responses with a smiley emoji.",
        ),
    ]

    config = deepcopy(CONFIG)
    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="How do I set it and what is the default value?",
        config=config,
        chat_history=prior_history,
        loaded_modules=[
            "hyperforge_rephrase",
            "hyperforge_static",
            "hyperforge_summarize",
        ],
    )
    # Check that max_tokens is in the rephrased question, which is the second rephrase step
    rephrase_steps = [
        step for step in question_memory.steps if step.module == "rephrase"
    ]
    assert len(rephrase_steps) >= 2
    rephrase_step = rephrase_steps[1]
    assert "max_tokens" in rephrase_step.value
    assert question_memory.final_answer
    assert (
        "1024" in question_memory.final_answer and "😊" in question_memory.final_answer
    )
