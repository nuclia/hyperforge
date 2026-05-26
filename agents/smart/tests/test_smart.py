from copy import deepcopy
from unittest.mock import patch

import pytest
from hyperforge.engine import main as arag_main
from hyperforge.interaction import AragAnswer
from hyperforge.pubsub import UserToAgentInteraction
from tests.arag import NUA_KEY

DOG_AGENT_ID = "269192da-f5b6-4196-8c9a-bd77e32f237c"
hyperforge_ID = "1d372a1a-8aaf-4dc7-bcf2-b8393751beba"
PERPLEXITY_AGENT_ID = "0f5487c8-432d-4205-b5e0-d5c06b7a638f"

CONFIG = {
    "drivers": [
        {
            "provider": "perplexity",
            "identifier": "perplexity-01",
            "name": "perplexity",
            "config": {
                "key": "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",
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
    "preprocess": [],
    "context": [
        {
            "module": "smart",
            "title": "",
            "planning_mode": "plan_execute",
            "extra_prompt": "If the question is about Snoopy, use the dog context agent. If the question is about RAO, use the RAO context agent. for general questions use the perplexity agent. Always try to be as thorough as possible",
            "registered_agents": [
                {
                    "id": DOG_AGENT_ID,
                    "module": "static",
                    "title": "Static Agent",
                    "context": "Snoopy is Carmen's dog. He is small and very cute but a bit moody. He was a rescue dog. Carmen works at Progress Software, a company that provides software for developing and managing applications.",
                },
                {
                    "id": hyperforge_ID,
                    "module": "static",
                    "title": "Static Agent",
                    "context": "RAO is a product of Progress agentic RAG, it is a powerful agent orchestrator that you are using right now.",
                },
                {
                    "id": PERPLEXITY_AGENT_ID,
                    "module": "perplexity",
                    "source": "perplexity-01",
                    "title": "Perplexity Agent",
                },
            ],
            "registered_agents_descriptions": {
                DOG_AGENT_ID: "Provides information about Snoopy, Carmen's dog.",
                hyperforge_ID: "Provides information about RAO, the agentic RAG orchestrator.",
                PERPLEXITY_AGENT_ID: "A general-purpose agent that can answer questions using the Perplexity API.",
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
@pytest.mark.parametrize(
    "executor_model,planner_model",
    [
        pytest.param(None, None, id="default"),
        pytest.param("gemini-2.5-flash", "gemini-2.5-flash", id="gemini"),
    ],
)
async def test_smart(executor_model, planner_model):
    config = deepcopy(CONFIG)
    if executor_model:
        config["context"][0]["executor_model"] = executor_model
    if planner_model:
        config["context"][0]["planner_model"] = planner_model

    answers = []

    async def callback(obj: AragAnswer):
        answers.append(obj)

    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="Who is Snoopy?",
        config=config,
        callback=callback,
    )

    keywords = ["Snoopy", "Carmen", "dog"]

    assert question_memory.final_answer
    assert all(
        keyword.lower() in question_memory.final_answer.lower() for keyword in keywords
    )

    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="What is RAO?",
        config=config,
        callback=callback,
    )

    keywords = ["RAO", "powerful", "agent", "orchestrator"]

    assert question_memory.final_answer
    assert all(
        keyword.lower() in question_memory.final_answer.lower() for keyword in keywords
    )


@pytest.mark.asyncio
@pytest.mark.vcr(ignore_localhost=True)
async def test_smart_parameters():
    question = "Who is Snoopy and can you tell me the stock price of the company where his owner works?"
    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question=question,
        config=CONFIG,
    )
    keywords = [
        "Snoopy",
        "Carmen",
        "dog",
        "Progress",
        "stock",
    ]
    assert question_memory.final_answer
    assert all(
        keyword.lower() in question_memory.final_answer.lower() for keyword in keywords
    )
    # Verify the planner selected dog-agent and  perplexity-agent, not rao-agent
    reactive_steps = [
        step
        for step in question_memory.steps
        if step.module == "smart" in step.title.lower()
    ]
    selected_tools = " ".join(step.value for step in reactive_steps)
    assert DOG_AGENT_ID in selected_tools, (
        f"Expected dog-agent in selected tools: {selected_tools}"
    )
    assert PERPLEXITY_AGENT_ID in selected_tools, (
        f"Expected perplexity-agent in selected tools: {selected_tools}"
    )
    assert hyperforge_ID not in selected_tools, (
        f"rao-agent should not have been called: {selected_tools}"
    )

    # Verify that the planning mode also works correctly
    config = deepcopy(CONFIG)
    config["context"][0]["planning_mode"] = "plan_execute"
    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question=question,
        config=config,
    )
    assert question_memory.final_answer
    assert all(
        keyword.lower() in question_memory.final_answer.lower() for keyword in keywords
    )
    planning_steps = [
        step
        for step in question_memory.steps
        if step.module == "smart" and "planner" in step.title.lower()
    ]
    assert planning_steps, "Expected at least one planning step"
    executing_steps = [
        step
        for step in question_memory.steps
        if step.module == "smart" and "executor" in step.title.lower()
    ]
    assert executing_steps, "Expected at least one executing step"
    selected_tools = " ".join(step.value for step in executing_steps)
    assert DOG_AGENT_ID in selected_tools, (
        f"Expected dog-agent in selected tools: {selected_tools}"
    )
    assert PERPLEXITY_AGENT_ID in selected_tools, (
        f"Expected perplexity-agent in selected tools: {selected_tools}"
    )
    assert hyperforge_ID not in selected_tools, (
        f"rao-agent should not have been called: {selected_tools}"
    )


@pytest.mark.asyncio
@pytest.mark.vcr(ignore_localhost=True)
async def test_smart_calls_correct_agent():
    """Verify the smart agent calls only the relevant agent for a focused question."""
    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="Is Snoopy a rescue dog?",
        config=CONFIG,
    )

    assert question_memory.final_answer
    assert "rescue" in question_memory.final_answer.lower()
    # Verify the planner selected dog-agent, not rao-agent
    smart_steps = [
        step
        for step in question_memory.steps
        if step.module == "smart" in step.title.lower()
    ]
    assert smart_steps, "Expected at least two smart agent steps"
    selected_tools = " ".join(step.value for step in smart_steps)
    assert DOG_AGENT_ID in selected_tools, (
        f"Expected dog-agent in selected tools: {selected_tools}"
    )
    assert hyperforge_ID not in selected_tools, (
        f"rao-agent should not have been called: {selected_tools}"
    )


@pytest.mark.asyncio
@pytest.mark.vcr(ignore_localhost=True)
async def test_smart_with_history():
    """Verify the smart agent includes session history when history is enabled."""
    config = deepcopy(CONFIG)
    config["context"][0]["history"] = True
    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="Who is Snoopy?",
        config=config,
    )

    assert question_memory.final_answer
    assert any(
        keyword.lower() in question_memory.final_answer.lower()
        for keyword in ["Snoopy", "Carmen", "dog"]
    )
    # Verify a history check step was recorded
    history_steps = [
        step
        for step in question_memory.steps
        if step.module == "smart" and "history" in step.title.lower()
    ]
    assert history_steps, "Expected a history check step when history is enabled"


@pytest.mark.asyncio
@pytest.mark.vcr(ignore_localhost=True)
async def test_smart_with_user_feedback():
    """Verify the smart agent uses the user_feedback tool when enabled and incorporates user responses."""
    config = deepcopy(CONFIG)
    config["context"][0]["enable_user_feedback"] = True
    config["context"][0]["feedback_timeout"] = 5_000
    config["context"][0]["extra_prompt"] = (
        "If the question is about Snoopy, ask the user whether it is a pet or the cartoon. If the question is about Snoopy the pet, use the dog context agent. If the question is about RAO, use the RAO context agent. for general questions use the perplexity agent. Always try to be as thorough as possible"
    )

    answers: list[AragAnswer] = []

    async def callback(obj: AragAnswer):
        answers.append(obj)

    async def mock_send_feedback(feedback):
        return UserToAgentInteraction(
            request_id=feedback.request_id,
            response="Snoopy the pet",
        )

    with patch(
        "hyperforge.memory.memory.QuestionMemory.send_feedback",
        side_effect=mock_send_feedback,
    ):
        question_memory = await arag_main(
            agent_id="default",
            internal_nua=False,
            external_nua_api_key=NUA_KEY,
            question="Tell me about Snoopy",
            config=config,
            callback=callback,
        )

    assert question_memory.final_answer
    assert "Feedback response: Snoopy the pet" in question_memory.steps[1].value
    assert any(
        keyword.lower() in question_memory.final_answer.lower()
        for keyword in ["Snoopy", "dog", "Carmen"]
    )
