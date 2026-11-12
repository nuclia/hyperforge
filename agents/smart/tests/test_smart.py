import os
from copy import deepcopy
from dataclasses import dataclass
from types import MethodType, SimpleNamespace
from typing import Any, ClassVar, cast
from unittest.mock import patch

import pytest
from hyperforge.context.agent import ContextAgent
from hyperforge.definition import FunctionDefinition
from hyperforge.engine import main as hyperforge_main
from hyperforge.interaction import AragAnswer
from hyperforge.manager import Manager
from hyperforge.memory.memory import EphemeralSessionMemory, MemoryConfig
from hyperforge.minimal_fixtures import cassette_nua_key
from hyperforge.models import Context, HistoryQuestionAnswer, Rules
from hyperforge.pubsub import UserToAgentInteraction
from nuclia.lib.nua_responses import Author, Message, ToolChoiceAuto

from hyperforge_smart.agent import RegisteredAgent, SmartAgent
from hyperforge_smart.config import SmartAgentConfig

NUA_KEY = os.environ.get(
    "NUA_KEY",
) or cassette_nua_key("https://europe-1.dp.progress.cloud/")

PERPLEXITY_KEY = os.environ.get("PERPLEXITY_API_KEY", "DUMMY_PERPLEXITY_KEY")

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
                "key": PERPLEXITY_KEY,
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
async def test_plan_executor_preserves_messages_between_iterations():
    observed_messages = []

    async def choose_tools(_self, *, messages, **_kwargs):
        observed_messages.append(messages)
        return SimpleNamespace(tools=None), 0, 0

    smart: Any = SimpleNamespace(
        config=SimpleNamespace(extra_prompt=None, max_iterations=1),
        choose_tools=MethodType(choose_tools, object()),
        _planner_attempts_from_registry=SmartAgent._planner_attempts_from_registry,
    )
    memory: Any = SimpleNamespace(get_tracking_info=lambda: None)
    manager: Any = None
    messages = [
        Message(author=Author.USER, text="Coordinate the incident"),
        Message(author=Author.NUCLIA, text="Tour Manager approved the change"),
    ]

    await SmartAgent._call_executor(
        smart,
        memory=memory,
        manager=manager,
        question="Coordinate the incident",
        steps=[{"description": "Continue coordination", "reason": "Approval received"}],
        summary="The Tour Manager approved the change",
        tools=[],
        messages=messages,
        resolved_feedback={},
    )

    assert observed_messages == [messages]
    assert observed_messages[0] is messages
    assert observed_messages[0][-1].text == "Tour Manager approved the change"


@pytest.mark.asyncio
async def test_feedback_decision_id_survives_rephrasing():
    class Memory:
        calls = 0

        def get_session_id(self):
            return "session-1"

        async def send_feedback(self, feedback):
            self.calls += 1
            return UserToAgentInteraction(
                request_id=feedback.request_id,
                response="Approved",
            )

        async def add_step(self, **_kwargs):
            return None

    memory: Any = Memory()
    manager: Any = None
    smart: Any = SimpleNamespace(
        config=SimpleNamespace(
            id="operations",
            module="smart",
            feedback_timeout=5_000,
            max_tool_result_bytes=None,
            max_tool_result_item_bytes=None,
        ),
        step_title=lambda title: title,
    )
    smart._process_results = MethodType(SmartAgent._process_results, smart)
    smart._inspect_result = MethodType(SmartAgent._inspect_result, smart)
    messages: list[Message] = []
    resolved_feedback: dict[str, str] = {}

    for question in (
        "Approve the shorter set?",
        "Can we reduce the set to 60 minutes?",
    ):
        await SmartAgent._execute_tool_calls_turn(
            smart,
            memory=memory,
            manager=manager,
            messages=messages,
            tool_calls=[
                (
                    "user_feedback",
                    {"decision_id": "set-duration", "question": question},
                )
            ],
            turn_label="test",
            resolved_feedback=resolved_feedback,
        )

    assert memory.calls == 1
    assert resolved_feedback == {"set-duration": "Approved"}
    assert "Decision ID: set-duration" in messages[-1].text


@pytest.mark.asyncio
async def test_different_feedback_decisions_are_both_requested():
    class Memory:
        calls = 0

        def get_session_id(self):
            return "session-1"

        async def send_feedback(self, feedback):
            self.calls += 1
            return UserToAgentInteraction(
                request_id=feedback.request_id,
                response="Approved",
            )

        async def add_step(self, **_kwargs):
            return None

    memory: Any = Memory()
    manager: Any = None
    smart: Any = SimpleNamespace(
        config=SimpleNamespace(
            id="operations",
            module="smart",
            feedback_timeout=5_000,
            max_tool_result_bytes=None,
            max_tool_result_item_bytes=None,
        ),
        step_title=lambda title: title,
    )
    smart._process_results = MethodType(SmartAgent._process_results, smart)
    smart._inspect_result = MethodType(SmartAgent._inspect_result, smart)
    messages: list[Message] = []
    resolved_feedback: dict[str, str] = {}

    for decision_id in ("set-duration", "gate-activation"):
        await SmartAgent._execute_tool_calls_turn(
            smart,
            memory=memory,
            manager=manager,
            messages=messages,
            tool_calls=[
                (
                    "user_feedback",
                    {"decision_id": decision_id, "question": f"Approve {decision_id}?"},
                )
            ],
            turn_label="test",
            resolved_feedback=resolved_feedback,
        )

    assert memory.calls == 2
    assert resolved_feedback == {
        "set-duration": "Approved",
        "gate-activation": "Approved",
    }


@pytest.mark.asyncio
async def test_feedback_response_with_upstream_request_id_is_cached():
    class Memory:
        def get_session_id(self):
            return "session-1"

        async def send_feedback(self, _feedback):
            return UserToAgentInteraction(
                request_id="upstream-a2a-request", response="Approved"
            )

        async def add_step(self, **_kwargs):
            return None

    smart: Any = SimpleNamespace(
        config=SimpleNamespace(
            id="operations",
            module="smart",
            feedback_timeout=5_000,
            max_tool_result_bytes=None,
            max_tool_result_item_bytes=None,
        ),
        step_title=lambda title: title,
    )
    smart._process_results = MethodType(SmartAgent._process_results, smart)
    smart._inspect_result = MethodType(SmartAgent._inspect_result, smart)
    resolved_feedback: dict[str, str] = {}
    memory: Any = Memory()
    manager: Any = None

    await SmartAgent._execute_tool_calls_turn(
        smart,
        memory=memory,
        manager=manager,
        messages=[],
        tool_calls=[
            (
                "user_feedback",
                {"decision_id": "set-duration", "question": "Approve?"},
            )
        ],
        turn_label="test",
        resolved_feedback=resolved_feedback,
    )

    assert resolved_feedback == {"set-duration": "Approved"}


@pytest.mark.asyncio
async def test_cached_feedback_does_not_skip_other_tools_in_same_turn():
    class Memory:
        async def add_step(self, **_kwargs):
            return None

    async def execute_tool_call(_memory, _manager, name, _args):
        return name, "tool result"

    smart: Any = SimpleNamespace(
        config=SimpleNamespace(
            id="operations",
            module="smart",
            parallel_tool_calls=False,
            max_tool_result_bytes=None,
            max_tool_result_item_bytes=None,
        ),
        step_title=lambda title: title,
        execute_tool_call=execute_tool_call,
    )
    smart._process_results = MethodType(SmartAgent._process_results, smart)
    smart._inspect_result = MethodType(SmartAgent._inspect_result, smart)
    smart._tool_call_key = SmartAgent._tool_call_key
    smart._classify_tool_result = SmartAgent._classify_tool_result
    memory: Any = Memory()
    manager: Any = None

    results = await SmartAgent._execute_tool_calls_turn(
        smart,
        memory=memory,
        manager=manager,
        messages=[],
        tool_calls=[
            (
                "user_feedback",
                {"decision_id": "set-duration", "question": "Reduce the set?"},
            ),
            ("venue_lookup", {"gate": "C"}),
        ],
        turn_label="test",
        resolved_feedback={"set-duration": "Approved"},
    )

    assert [name for name, _ in results] == ["user_feedback", "venue_lookup"]
    assert results[1] == ("venue_lookup", "tool result")


@pytest.mark.asyncio
async def test_executor_runs_tools_returned_with_task_complete():
    smart: Any = SimpleNamespace(
        config=SimpleNamespace(max_iterations=1, extra_prompt=None),
    )

    async def choose_tools(**_kwargs):
        return (
            SimpleNamespace(
                tools={
                    "calls": [
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="a2a_query_of_logistics", arguments={}
                            )
                        ),
                        SimpleNamespace(
                            function=SimpleNamespace(name="task_complete", arguments={})
                        ),
                    ]
                }
            ),
            0,
            0,
        )

    executed_calls = []

    async def execute_turn(**kwargs):
        executed_calls.extend(kwargs["tool_calls"])
        return [("a2a_query of logistics", "result")]

    smart.choose_tools = choose_tools
    smart._execute_tool_calls_turn = execute_turn
    smart._planner_attempts_from_registry = SmartAgent._planner_attempts_from_registry
    memory: Any = SimpleNamespace(get_tracking_info=lambda: None)
    manager: Any = None

    results, _, _, _ = await SmartAgent._call_executor(
        smart,
        memory=memory,
        manager=manager,
        question="question",
        steps=[],
        summary="",
        tools=[],
        messages=[],
        resolved_feedback={},
    )

    assert executed_calls == [("a2a_query_of_logistics", {})]
    assert results == [("a2a_query of logistics", "result")]


def test_new_feedback_prevents_same_turn_completion():
    calls = [
        ("user_feedback", {"decision_id": "approval", "question": "Approve?"}),
        ("task_complete", {}),
    ]

    assert SmartAgent._has_unresolved_feedback(calls, {})
    assert not SmartAgent._has_unresolved_feedback(calls, {"approval": "yes"})


@pytest.mark.asyncio
async def test_choose_tools_uses_auto_tool_choice():
    smart_agent = SmartAgent(
        config=SmartAgentConfig.model_validate(
            {
                "id": "smart-test",
                "module": "smart",
                "title": "Smart Agent",
                "planning_mode": "reactive",
            }
        )
    )
    captured_items = []

    async def execute_raw(item, tracking=None):
        captured_items.append(item)
        return SimpleNamespace(), 0.0, 0.0

    await smart_agent.choose_tools(
        manager=SimpleNamespace(execute_raw=execute_raw),
        messages=[],
        tools=[],
    )

    assert isinstance(captured_items[0].tool_choice, ToolChoiceAuto)


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

    question_memory = await hyperforge_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="Who is Snoopy?",
        config=config,
        callback=callback,
        loaded_modules=[
            "hyperforge_smart",
            "hyperforge_static",
            "hyperforge_perplexity",
            "hyperforge_summarize",
        ],
    )

    keywords = ["Snoopy", "Carmen", "dog"]

    assert question_memory.final_answer
    assert all(
        keyword.lower() in question_memory.final_answer.lower() for keyword in keywords
    )

    question_memory = await hyperforge_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="What is RAO?",
        config=config,
        callback=callback,
        loaded_modules=[
            "hyperforge_smart",
            "hyperforge_static",
            "hyperforge_perplexity",
            "hyperforge_summarize",
        ],
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
    question_memory = await hyperforge_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question=question,
        config=CONFIG,
        loaded_modules=[
            "hyperforge_smart",
            "hyperforge_static",
            "hyperforge_perplexity",
            "hyperforge_summarize",
        ],
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
    question_memory = await hyperforge_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question=question,
        config=config,
        loaded_modules=[
            "hyperforge_smart",
            "hyperforge_static",
            "hyperforge_perplexity",
            "hyperforge_summarize",
        ],
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
    question_memory = await hyperforge_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="Is Snoopy a rescue dog?",
        config=CONFIG,
        loaded_modules=[
            "hyperforge_smart",
            "hyperforge_static",
            "hyperforge_perplexity",
            "hyperforge_summarize",
        ],
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
    question_memory = await hyperforge_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="Who is Snoopy?",
        config=config,
        loaded_modules=[
            "hyperforge_smart",
            "hyperforge_static",
            "hyperforge_perplexity",
            "hyperforge_summarize",
        ],
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
            request_id="upstream-a2a-request",
            response="Snoopy the pet",
        )

    with patch(
        "hyperforge.memory.memory.QuestionMemory.send_feedback",
        side_effect=mock_send_feedback,
    ):
        question_memory = await hyperforge_main(
            agent_id="default",
            internal_nua=False,
            external_nua_api_key=NUA_KEY,
            question="Tell me about Snoopy",
            config=config,
            callback=callback,
            loaded_modules=[
                "hyperforge_smart",
                "hyperforge_static",
                "hyperforge_perplexity",
                "hyperforge_summarize",
            ],
        )

    assert question_memory.final_answer
    feedback_steps = [
        step
        for step in question_memory.steps
        if step.module == "smart" and "user feedback" in step.title.lower()
    ]
    assert feedback_steps, "Expected a user feedback step"
    assert any(
        "Feedback response: Snoopy the pet" in (step.value or "")
        for step in feedback_steps
    )
    assert any(
        keyword.lower() in question_memory.final_answer.lower()
        for keyword in ["Snoopy", "dog", "Carmen"]
    )


@dataclass
class SpyReactiveMemory:
    question_memory: object
    context_history_calls: int = 0
    get_chat_history_calls: int = 0

    def __getattr__(self, name):
        return getattr(self.question_memory, name)

    async def context_history(self):
        self.context_history_calls += 1
        return (
            "User: Earlier question about Snoopy\nAssistant: Earlier answer about Snoopy",
            1,
        )

    async def get_chat_history(self):
        self.get_chat_history_calls += 1
        return [
            Message(author=Author.USER, text="Earlier question about Snoopy"),
            Message(author=Author.NUCLIA, text="Earlier answer about Snoopy"),
        ]


class EmptyResultAgent(ContextAgent):
    __published_functions__: ClassVar[dict[str, FunctionDefinition]] = {
        "lookup": FunctionDefinition(
            name="lookup",
            description="Return context for a lookup request.",
            parameters={
                "query": {
                    "type": "string",
                    "description": "Query to look up.",
                }
            },
        )
    }

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.calls: list[dict[str, str]] = []

    async def preload(self, manager, memory):
        return None

    async def lookup(self, memory, manager, query: str):
        self.calls.append({"query": query})
        return Context(
            original_question_uuid=memory.original_question_uuid,
            actual_question_uuid=memory.actual_question_uuid,
            question=query,
            source="dummy",
            agent="dummy",
            agent_id=self.agent_id,
            title="Empty result",
            chunks=[],
        )


class EmptyLabelsAgent(ContextAgent):
    __published_functions__: ClassVar[dict[str, FunctionDefinition]] = {
        "labels": FunctionDefinition(
            name="labels",
            description="Return no labels.",
            parameters={},
        )
    }

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.calls = 0

    async def preload(self, manager, memory):
        return None

    async def labels(self, memory, manager):
        self.calls += 1
        return {}


def make_tool_response(name: str, arguments: dict[str, str]):
    return SimpleNamespace(
        tools={
            name: [
                SimpleNamespace(
                    function=SimpleNamespace(name=name, arguments=arguments)
                )
            ]
        }
    )


@pytest.mark.asyncio
async def test_smart_reactive_uses_structured_chat_history_messages():
    smart_agent = SmartAgent(
        config=SmartAgentConfig.model_validate(
            {
                "id": "smart-test",
                "module": "smart",
                "title": "Smart Agent",
                "planning_mode": "reactive",
                "history": True,
                "max_iterations": 1,
            }
        )
    )
    smart_agent.registered_agents = []

    memory = EphemeralSessionMemory.from_config(
        MemoryConfig.model_validate({}),
        agent_id="agent",
        workflow_id="default",
        rules=Rules(rules=[]),
    )
    memory.init("session")
    base_question_memory = memory.start_question(
        "Use previous Snoopy context",
        question_id="question-id",
        chat_history=[
            HistoryQuestionAnswer(
                question="Earlier question about Snoopy",
                answer="Earlier answer about Snoopy",
            )
        ],
    )
    question_memory = SpyReactiveMemory(base_question_memory)

    captured_messages = []

    async def fake_choose_tools(
        self, manager, messages, tools, system_override=None, tracking=None
    ):
        captured_messages.extend(messages)
        return make_tool_response("task_complete", {}), 0.0, 0.0

    with patch.object(SmartAgent, "choose_tools", fake_choose_tools):
        await smart_agent.smart_planner(
            question="Use previous Snoopy context",
            memory=cast(Any, question_memory),
            manager=cast(Manager, SimpleNamespace(spec=Manager)),
            question_uuid="question-id",
        )

    assert question_memory.context_history_calls == 0
    assert question_memory.get_chat_history_calls == 1
    assert [message.author for message in captured_messages] == [
        Author.USER,
        Author.NUCLIA,
        Author.USER,
    ]
    assert captured_messages[0].text == "Earlier question about Snoopy"
    assert captured_messages[1].text == "Earlier answer about Snoopy"
    assert captured_messages[2].text == "Use previous Snoopy context"


@pytest.mark.asyncio
async def test_smart_reactive_does_not_repeat_identical_empty_tool_call():
    agent_id = "empty-agent"
    empty_agent = EmptyResultAgent(agent_id=agent_id)
    smart_agent = SmartAgent(
        config=SmartAgentConfig.model_validate(
            {
                "id": "smart-test",
                "module": "smart",
                "title": "Smart Agent",
                "planning_mode": "reactive",
                "max_iterations": 3,
            }
        )
    )
    smart_agent.registered_agents = [
        RegisteredAgent(
            agent=empty_agent,
            description="Returns an empty result for testing.",
            available_functions=empty_agent.__published_functions__,
        )
    ]

    memory = EphemeralSessionMemory.from_config(
        MemoryConfig.model_validate({}),
        agent_id="agent",
        workflow_id="default",
        rules=Rules(rules=[]),
    )
    memory.init("session")
    question_memory = memory.start_question("Find Snoopy", question_id="question-id")

    responses = iter(
        [
            (make_tool_response(f"lookup__{agent_id}", {"query": "Snoopy"}), 0.0, 0.0),
            (make_tool_response(f"lookup__{agent_id}", {"query": "Snoopy"}), 0.0, 0.0),
            (make_tool_response("task_complete", {}), 0.0, 0.0),
        ]
    )

    async def fake_choose_tools(
        self, manager, messages, tools, system_override=None, tracking=None
    ):
        return next(responses)

    with patch.object(SmartAgent, "choose_tools", fake_choose_tools):
        await smart_agent.smart_planner(
            question="Find Snoopy",
            memory=question_memory,
            manager=SimpleNamespace(spec=Manager),
            question_uuid="question-id",
        )

    assert empty_agent.calls == [{"query": "Snoopy"}]


@pytest.mark.asyncio
async def test_smart_reactive_hides_empty_parameterless_tool():
    agent_id = "empty-agent"
    empty_agent = EmptyLabelsAgent(agent_id=agent_id)
    smart_agent = SmartAgent(
        config=SmartAgentConfig.model_validate(
            {
                "id": "smart-test",
                "module": "smart",
                "title": "Smart Agent",
                "planning_mode": "reactive",
                "max_iterations": 2,
            }
        )
    )
    smart_agent.registered_agents = [
        RegisteredAgent(
            agent=empty_agent,
            description="Returns an empty result for testing.",
            available_functions=empty_agent.__published_functions__,
        )
    ]

    memory = EphemeralSessionMemory.from_config(
        MemoryConfig.model_validate({}),
        agent_id="agent",
        workflow_id="default",
        rules=Rules(rules=[]),
    )
    memory.init("session")
    question_memory = memory.start_question("Find Snoopy", question_id="question-id")

    responses = iter(
        [
            (make_tool_response(f"labels__{agent_id}", {}), 0.0, 0.0),
            (make_tool_response("task_complete", {}), 0.0, 0.0),
        ]
    )
    tools_by_turn = []

    async def fake_choose_tools(
        self, manager, messages, tools, system_override=None, tracking=None
    ):
        tools_by_turn.append([tool.name for tool in tools])
        return next(responses)

    with patch.object(SmartAgent, "choose_tools", fake_choose_tools):
        await smart_agent.smart_planner(
            question="Find Snoopy",
            memory=question_memory,
            manager=SimpleNamespace(spec=Manager),
            question_uuid="question-id",
        )

    assert empty_agent.calls == 1
    assert f"labels__{agent_id}" in tools_by_turn[0]
    assert f"labels__{agent_id}" not in tools_by_turn[1]


@pytest.mark.asyncio
async def test_smart_plan_execute_surfaces_empty_attempts_to_planner():
    agent_id = "empty-agent"
    empty_agent = EmptyResultAgent(agent_id=agent_id)
    smart_agent = SmartAgent(
        config=SmartAgentConfig.model_validate(
            {
                "id": "smart-test",
                "module": "smart",
                "title": "Smart Agent",
                "planning_mode": "plan_execute",
                "max_iterations": 3,
            }
        )
    )
    smart_agent.registered_agents = [
        RegisteredAgent(
            agent=empty_agent,
            description="Returns an empty result for testing.",
            available_functions=empty_agent.__published_functions__,
        )
    ]

    memory = EphemeralSessionMemory.from_config(
        MemoryConfig.model_validate({}),
        agent_id="agent",
        workflow_id="default",
        rules=Rules(rules=[]),
    )
    memory.init("session")
    question_memory = memory.start_question("Find Snoopy", question_id="question-id")

    responses = iter(
        [
            (make_tool_response(f"lookup__{agent_id}", {"query": "Snoopy"}), 0.0, 0.0),
            (make_tool_response("task_complete", {}), 0.0, 0.0),
        ]
    )
    planner_prompts: list[str] = []

    async def fake_choose_tools(
        self, manager, messages, tools, system_override=None, tracking=None
    ):
        return next(responses)

    async def fake_execute_json(*args, **kwargs):
        prompt = kwargs["prompt"]
        planner_prompts.append(prompt)
        if len(planner_prompts) == 1:
            return (
                {
                    "status": "plan",
                    "reasoning": "Need to search once.",
                    "summary": "Nothing yet.",
                    "steps": [
                        {
                            "description": "Look up Snoopy in the available source.",
                            "reason": "Need source-specific context.",
                        }
                    ],
                },
                0.0,
                0.0,
            )

        assert "**Tool attempts:**" in prompt
        assert f"- lookup__{agent_id}(" in prompt
        assert ": empty" in prompt and "Snoopy" in prompt
        return (
            {
                "status": "done",
                "reasoning": "The previous attempt was exhausted and should not be repeated.",
                "summary": "The lookup path returned no useful information.",
                "steps": [],
            },
            0.0,
            0.0,
        )

    manager = SimpleNamespace(execute_json=fake_execute_json)

    with patch.object(SmartAgent, "choose_tools", fake_choose_tools):
        await smart_agent.smart_planner(
            question="Find Snoopy",
            memory=question_memory,
            manager=manager,
            question_uuid="question-id",
        )

    assert empty_agent.calls == [{"query": "Snoopy"}]
    assert len(planner_prompts) == 2
