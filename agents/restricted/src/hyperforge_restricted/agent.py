from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Optional, cast

from hyperforge.agent import Agent
from hyperforge.configure import agent
from hyperforge.context.agent import ContextAgent, build_context_agent
from hyperforge.db.agents import SERVICE_NAME
from hyperforge.definition import FunctionDefinition
from hyperforge.interaction import Feedback
from hyperforge.manager import Manager
from hyperforge.memory import Context, QuestionMemory
from hyperforge_conditional.context_agent import (
    ContextConditional,
    ContextConditionalAgentConfig,
)
from hyperforge_rephrase.agent import RephraseAgent
from hyperforge_rephrase.config import RephraseAgentConfig
from nuclia_models.predict.remi import RemiResponse
from nucliadb_telemetry.utils import get_telemetry
from opentelemetry import trace
from RestrictedPython.Guards import safe_builtins  # type: ignore

from hyperforge import logger
from hyperforge_restricted.config import PythonAgentConfig
from hyperforge_restricted.decision import (
    CHOOSE_AGENT_TEMPLATE,
    CHOOSE_SCHEMA,
    EXTRACT_AGENT_TEMPLATE,
    TRANSFORM_REPHRASE,
)
from hyperforge_restricted.model import (
    RestrictedPythonTask,
    WorkerExecutionRequest,
    WorkerTypes,
)

from .sandbox import SandboxRunner
from .sandbox import settings as sandbox_settings


def tracer():
    provider = get_telemetry(SERVICE_NAME)
    if provider:
        return provider.get_tracer(__name__)
    else:
        return trace.NoOpTracer()


# Define allowed builtins and imports
allowed_builtins = {
    **safe_builtins,  # Default safe builtins
    "__import__": lambda name, *args: (
        __import__(name) if name in {"math"} else None
    ),  # Restrict imports
}

# Define restricted globals
restricted_globals = {
    "__builtins__": allowed_builtins,  # Restrict built-ins
}


@dataclass
class Message:
    function: str
    agent: str


@dataclass
class Output:
    resolved: bool = False
    missing: List[str] = field(default_factory=list)
    contexts: List[Context] = field(default_factory=list)
    final_answer: Optional[str] = None


if sandbox_settings.sandbox_socket is None:
    import os

    def init_sandbox():
        # Set lowest priority
        os.nice(19)

    SANDBOX_POOL = ProcessPoolExecutor(10, initializer=init_sandbox)


@agent(
    id="restricted",
    agent_type="context",
    title="Python (Restricted) Context Agent",
    description="Use Python code in a restricted environment to gather context to answer questions.",
    config_schema=PythonAgentConfig,
)
class PythonAgent(Agent[PythonAgentConfig], ContextAgent):
    flow_id: Optional[str] = None
    context: Optional[Context] = None
    agents: Dict[str, "ContextAgent"]
    function_names: Dict[str, Dict[str, FunctionDefinition]]
    output: Output

    def __init__(
        self, config: PythonAgentConfig, agent_id: Optional[str] = None
    ) -> None:
        super().__init__(config, agent_id)
        self.agents = {}
        self.function_names = {}

    async def inner_from_config(
        self, config: PythonAgentConfig, agent_id: Optional[str] = None
    ):
        await self.context_from_config(config)
        agents = [
            await build_context_agent(sub_agent_config)
            for sub_agent_config in config.agents or []
        ]
        if agents is None or len(agents) == 0:
            raise ValueError("At least one agent must be provided in agents list")

        agents_dict = {
            child_agent.agent_id: child_agent
            for child_agent in agents
            if child_agent is not None
        }

        function_names = {
            child_agent.agent_id: child_agent.__published_functions__
            for child_agent in agents
            if child_agent is not None
        }

        function_names["self"] = {
            "transform": FunctionDefinition(
                name="transform",
                description="Use this function to transform the question or context in a custom way",
                parameters={
                    "question": {
                        "type": "string",
                        "description": "The question to be transformed",
                    },
                    "context": {
                        "type": "string",
                        "description": "The context to be transformed",
                    },
                },
            ),
            "rephrase": FunctionDefinition(
                name="rephrase",
                description="Use this function to rephrase the question based on context",
                parameters={
                    "question": {
                        "type": "string",
                        "description": "The question to be rephrased",
                    },
                    "model": {
                        "type": "string",
                        "description": "The model to use for rephrasing",
                    },
                    "session_info": {
                        "type": "boolean",
                        "description": "Indicates whether to include session information in the rephrasing",
                    },
                    "history": {
                        "type": "boolean",
                        "description": "Indicates whether to include conversation history in the rephrasing",
                    },
                    "split_question": {
                        "type": "boolean",
                        "description": "Indicates whether to split the question into sub-questions",
                    },
                    "context": {
                        "type": "string",
                        "description": "The context to use for rephrasing",
                    },
                    "provided_synonyms": {
                        "type": "object",
                        "description": "A dictionary of provided synonyms to use in the rephrasing",
                        "additionalProperties": {"type": "string"},
                    },
                    "rules": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                        "description": "A list of rules to follow when rephrasing the question",
                    },
                },
            ),
            "user_validation": FunctionDefinition(
                name="user_validation",
                description="Use this function to send a validation question to the user and get feedback",
                parameters={
                    "question": {
                        "type": "string",
                        "description": "The question to ask the user for validation",
                    },
                    "response_schema": {
                        "type": "object",
                        "description": "The schema defining the expected response from the user",
                    },
                },
            ),
            "add_step": FunctionDefinition(
                name="add_step",
                description="Use this function to add a step to the question memory",
                parameters={
                    "reason": {
                        "type": "string",
                        "description": "The reason for adding the step",
                    },
                    "value": {
                        "type": "string",
                        "description": "The value of the step",
                    },
                },
            ),
            "remi_validation": FunctionDefinition(
                name="remi_validation",
                description="Use this function to validate the context against the question using REMi model",
                parameters={
                    "context": {
                        "type": "object",
                        "description": "The context to be validated",
                    },
                    "question": {
                        "type": "string",
                        "description": "The question to validate against",
                    },
                    "answer": {
                        "type": "string",
                        "description": "The answer generated for the question (optional)",
                    },
                },
            ),
            "context_match_question": FunctionDefinition(
                name="context_match_question",
                description="Check if the provided context matches the question",
                parameters={
                    "context": {
                        "type": "object",
                        "description": "The context to be checked",
                    },
                    "question": {
                        "type": "string",
                        "description": "The question to be matched against",
                    },
                },
            ),
            "extract": FunctionDefinition(
                name="extract",
                description="Extract relevant information from the question based on the provided context",
                parameters={
                    "question": {
                        "type": "string",
                        "description": "The question to extract information from",
                    },
                    "what_to_extract": {
                        "type": "string",
                        "description": "Description of what information to extract",
                    },
                },
            ),
            "validate": FunctionDefinition(
                name="validate",
                description="Use this function to ask a question to the next agent in the chain",
                parameters={
                    "question": {
                        "type": "string",
                        "description": "The question to be asked to the next agent",
                    }
                },
            ),
            "choose": FunctionDefinition(
                name="choose",
                description="Choose from multiple intents the one that fits best the user's question",
                parameters={
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                        "description": "The options to choose from",
                    },
                    "extra_info": {
                        "type": "string",
                        "description": "Additional information to help with the choice",
                    },
                },
            ),
            "summary": FunctionDefinition(
                name="summary",
                description="Use this function to summarize multiple contexts into one",
                parameters={
                    "contexts": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                        "description": "The list of contexts to be summarized",
                    }
                },
            ),
            "set_context": FunctionDefinition(
                name="set_context",
                description="Use this function to set the context of the current agent",
                parameters={
                    "context": {
                        "type": "object",
                        "description": "The context to be set",
                    }
                },
            ),
        }

        self.agents = agents_dict
        self.function_names = function_names
        self.output = Output()
        await self.context_from_config(config)

    def _runner(self, memory: QuestionMemory, manager: Manager):
        if sandbox_settings.sandbox_socket is not None:
            # Remote mode, use a socket to communicate with the sandbox server
            return SandboxRunner.remote(
                sandbox_settings.sandbox_socket,
                functools.partial(self.handle_queue_item, manager, memory),
            )
        elif self.config.debug:
            # Debug mode, run in the same process (another thread)
            return SandboxRunner.with_pool(
                ThreadPoolExecutor(1),
                functools.partial(self.handle_queue_item, manager, memory),
                debug=True,
            )
        else:
            # Process pool mode, run in a separate process for isolation
            return SandboxRunner.with_pool(
                SANDBOX_POOL,
                functools.partial(self.handle_queue_item, manager, memory),
            )

    async def _preload_child_agents(
        self, manager: Manager, memory: QuestionMemory
    ) -> None:
        """Call ``preload`` on every child agent that supports it.

        Child agents whose ``__published_functions__`` dict is empty after
        ``inner_from_config`` (e.g. ``MCPAgent``) declare their tools at
        runtime via this hook.  After preloading, ``function_names`` is
        refreshed so the sandbox picks up the newly discovered functions.

        Agents that do not override ``preload`` (the base no-op) are unaffected.
        """
        for agent_id, child_agent in self.agents.items():
            await child_agent.preload(manager, memory)
            if (
                not self.function_names.get(agent_id)
                and child_agent.__published_functions__
            ):
                self.function_names[agent_id] = child_agent.__published_functions__

    async def _get_question_context(
        self,
        memory: QuestionMemory,
        manager: Manager,
        question_uuid: str,
        question: str,
        flow_id: str,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[str, str]]:
        t0 = time()
        self.flow_id = flow_id
        self.context = Context(
            agent_id=self.config.id if self.config.id else "default",
            original_question_uuid=memory.original_question_uuid,
            actual_question_uuid=question_uuid,
            question=question,
            source="python",
            agent="python",
            title=self.config.title,
        )

        global_vars: Dict[str, Any] = {}

        await self._preload_child_agents(manager, memory)
        for parameter, type_value in self.config.parameters.items():
            if f"X-RESTRICTED-{parameter.upper()}" in memory.headers:
                if type_value == "bool":
                    global_vars[f"param_{parameter}"] = memory.headers[
                        f"X-RESTRICTED-{parameter.upper()}"
                    ].lower() in ["1", "true", "yes"]
                elif type_value == "int":
                    global_vars[f"param_{parameter}"] = int(
                        memory.headers[f"X-RESTRICTED-{parameter.upper()}"]
                    )
                else:
                    global_vars[f"param_{parameter}"] = memory.headers[
                        f"X-RESTRICTED-{parameter.upper()}"
                    ]

        depth = 0
        code = self.config.code
        self.output = Output(missing=[question])
        while self.output.resolved is False and depth < self.config.max_retries:
            missing_questions = self.output.missing.copy()
            self.output.missing.clear()

            tasks = []
            for local_question in missing_questions:
                copy_of_global_vars = deepcopy(global_vars)
                local_vars: Dict[str, Any] = {}
                local_vars["question"] = local_question
                local_vars["depth"] = depth
                local_vars["needs_rephrase"] = (
                    True if depth == 0 and self.config.needs_rephrase else False
                )
                if self.config.debug:
                    logger.debug(
                        f"Executing restricted code, loop {depth}, question: {question}"
                    )

                tasks.append(
                    asyncio.create_task(
                        self._runner(memory, manager).run(
                            WorkerExecutionRequest(
                                code=code,
                                question=question,
                                local_vars=local_vars,
                                global_vars=copy_of_global_vars,
                                function_names=self.function_names,
                            )
                        )
                    )
                )
            await asyncio.gather(*tasks)
            depth += 1

        await memory.add_step(
            step_module=self.config.module,
            step_title=self.step_title("Context retrieval"),
            step_reason=f"""Contexts: {len(self.output.contexts)} Resolved:{self.output.resolved} Missing: {len(self.output.missing)} retries to resolve""",
            step_agent_path=f"/context/{self.config.id if self.config.id else 'default'}",
            step_value=" - ".join([x.summary for x in self.output.contexts]),
            timeit=time() - t0,
            input_nuclia_tokens=0.5,
            output_nuclia_tokens=0.5,
        )
        if self.output.resolved:
            for context in self.output.contexts:
                await memory.save_context(flow_id=flow_id, context=context)
                if len(context.image_urls) > 0:
                    logger.debug(
                        f"Context {context.agent_id} has {len(context.image_urls)} images"
                    )

                    await memory.save_image_urls(image_urls=context.image_urls)
            if self.output.final_answer is not None:
                memory.final_answer = self.output.final_answer

        return []

    async def remi_validation(
        self,
        manager: Manager,
        memory: QuestionMemory,
        question: Optional[str] = None,
        context: Optional[Context] = None,
        answer: Optional[str] = None,
    ) -> RemiResponse:
        t0 = time()
        contexts = []
        if context is not None:
            if context.summary and answer is None:
                answer = context.summary

            for chunk in context.chunks:
                contexts.append(chunk.render())
            for structured in context.structured:
                contexts.append(structured)

        validation = await manager.remi(
            question=question, contexts=contexts, answer=answer
        )
        await memory.add_step(
            step_module="remi_validation",
            step_title=self.step_title("REMi validation"),
            step_reason="",
            step_value=validation.model_dump_json(),
            timeit=time() - t0,
            input_nuclia_tokens=validation.consumption.normalized_tokens.input
            if validation.consumption
            else 0.0,
            output_nuclia_tokens=validation.consumption.normalized_tokens.output
            if validation.consumption
            else 0.0,
            step_agent_path=f"/context/{self.agent_id}",
        )
        return validation

    async def handle_queue_item(
        self, manager: Manager, memory: QuestionMemory, item: RestrictedPythonTask
    ) -> WorkerTypes:
        contexts: List[Context] = []

        with tracer().start_as_current_span(f"RestrictedAgent.{item.function}"):
            try:
                item.keyword_args.update(
                    {
                        "manager": manager,
                        "memory": memory,
                    }
                )

                match item.function:
                    case "user_validation":
                        # feedback is a string with the feedback message
                        question = item.keyword_args.get("question", "")
                        response_schema = item.keyword_args.get("response_schema", {})
                        data = item.keyword_args.get("data", {})
                        await memory.send_feedback(
                            feedback=Feedback(
                                question=question,
                                response_schema=response_schema,
                                request_id=memory.get_session_id(),
                                module=self.config.module,
                                agent_id=self.agent_id,
                                data=data,
                            )
                        )
                    case "transform":
                        t0 = time()
                        # pattern is a string to search in the question
                        question = item.keyword_args.get("question", "")
                        context = item.keyword_args.get("context", "")
                        # Simple transform: concatenate context and question
                        prompt = TRANSFORM_REPHRASE.format(
                            context=context, question=question
                        )
                        output_text, input, output, _ = await manager.execute(
                            prompt=prompt,
                            user_id="transform",
                            model=self.config.decision_model,
                            tracking=memory.get_tracking_info(),
                        )
                        await memory.add_step(
                            step_module="transform",
                            step_title=self.step_title("Transform question"),
                            step_reason=f"{question}: {context}",
                            step_value=output_text,
                            timeit=time() - t0,
                            input_nuclia_tokens=input,
                            output_nuclia_tokens=output,
                            step_agent_path=f"/context/{self.agent_id}",
                        )
                        return output_text
                    case "rephrase":
                        # question is the question to rephrase
                        question = item.keyword_args.get("question", "")
                        model = item.keyword_args.get("model", "chatgpt-azure-4o")
                        session_info = item.keyword_args.get("session_info", False)
                        history = item.keyword_args.get("history", False)
                        split_question = item.keyword_args.get("split_question", False)
                        context = item.keyword_args.get("context", "")
                        provided_synonyms = item.keyword_args.get(
                            "provided_synonyms", {}
                        )
                        rules = item.keyword_args.get("rules", [])
                        rephrase_agent = await RephraseAgent.from_config(
                            config=RephraseAgentConfig(
                                model=model,
                                kb=None,
                                rids=[],
                                labels=[],
                                synonyms=False,
                                provided_synonyms=provided_synonyms,
                                extend=True,
                                rules=rules,
                                session_info=session_info,
                                history=history,
                                split_question=split_question,
                            )
                        )
                        new_questions = await rephrase_agent.inner_rephrase(
                            memory=memory,
                            manager=manager,
                            question=question,
                            extra_rule=context,
                        )
                        return new_questions
                    case "add_step":
                        # feedback is a string with the feedback message
                        reason = item.keyword_args.get("reason", "")
                        value = item.keyword_args.get("value", "")
                        await memory.add_step(
                            step_module="add_step",
                            step_title=self.step_title("Feedback"),
                            step_reason=reason,
                            step_value=value,
                            timeit=0.0,
                            input_nuclia_tokens=0.0,
                            output_nuclia_tokens=0.0,
                            step_agent_path=f"/context/{self.agent_id}",
                        )

                    case "choose":
                        # options is a dict of option_name: description
                        t0 = time()
                        question = item.keyword_args.get("question", "")
                        options = item.keyword_args.get("options", {})
                        extra_info = item.keyword_args.get("extra_info", None)
                        schema = CHOOSE_SCHEMA.copy()
                        schema["parameters"]["properties"]["selected"]["enum"].extend(  # type: ignore
                            options.keys()
                        )  # type: ignore
                        prompt = CHOOSE_AGENT_TEMPLATE.render(
                            options=options, question=question, extra_info=extra_info
                        )

                        sources, input, output = await manager.execute_json(
                            prompt=prompt,
                            schema=CHOOSE_SCHEMA,
                            user_id="case_selection",
                            model=self.config.decision_model,
                            tracking=memory.get_tracking_info(),
                        )
                        case = sources.get("selected", "else")
                        reason = sources.get("reason")
                        await memory.add_step(
                            step_module="case_selection",
                            step_title=self.step_title("Case selection"),
                            step_reason=reason,
                            step_value=f"Selected case: {case}",
                            timeit=time() - t0,
                            input_nuclia_tokens=input,
                            output_nuclia_tokens=output,
                            step_agent_path=f"/context/{self.agent_id}",
                        )
                        return case
                    case "extract":
                        t0 = time()
                        # pattern is a string to search in the question
                        request = item.keyword_args.get("request", "")
                        schema = item.keyword_args.get("schema", "")
                        labels = item.keyword_args.get("labels", [])
                        prompt = EXTRACT_AGENT_TEMPLATE.render(
                            request=request, labels=labels
                        )

                        (extracted, input, output) = await manager.execute_json(
                            prompt=prompt,
                            schema=schema,
                            user_id="extract",
                            model=self.config.decision_model,
                            tracking=memory.get_tracking_info(),
                        )
                        await memory.add_step(
                            step_module="case_selection",
                            step_title=self.step_title("Extraction"),
                            step_reason="",
                            step_value=f"Extracted information: {extracted}",
                            timeit=time() - t0,
                            input_nuclia_tokens=input,
                            output_nuclia_tokens=output,
                            step_agent_path=f"/context/{self.config.id if self.config.id else 'default'}",
                        )
                        return extracted
                    case "remi_validation":
                        context = item.keyword_args.get("context")
                        question = item.keyword_args.get("question")
                        manager = cast(Manager, item.keyword_args.get("manager"))
                        memory = cast(QuestionMemory, item.keyword_args.get("memory"))
                        answer = item.keyword_args.get("answer", None)
                        return await self.remi_validation(
                            context=context,
                            question=question,
                            manager=manager,
                            memory=memory,
                            answer=answer,
                        )

                    case "validate":
                        conditional_agent = await ContextConditional.from_config(
                            ContextConditionalAgentConfig(
                                model=self.config.decision_model,
                            )
                        )
                        decision = await conditional_agent.make_decision(
                            *item.args,  # type: ignore
                            **item.keyword_args,
                        )
                        return decision
                    case "save":
                        # pattern is a string to search in the question
                        contexts = item.keyword_args.get("contexts", [])
                        resolved: bool = item.keyword_args.get("resolved", False)
                        missing: List[str] = item.keyword_args.get("missing", [])
                        self.output.contexts.extend(contexts)
                        if resolved is True:
                            self.output.resolved = resolved
                        self.output.missing = missing
                        self.output.final_answer = item.keyword_args.get(
                            "final_answer", None
                        )
                        return self.output.resolved
                    case "save_context":
                        # pattern is a string to search in the question
                        contexts = item.keyword_args.get("contexts", [])
                        for context in contexts:
                            if context is not None:
                                await memory.save_context(
                                    flow_id=self.flow_id or "", context=context
                                )
                        return None
                    case _:
                        # Placeholder for actual handling logic
                        item.keyword_args["flow_id"] = self.flow_id
                        return await getattr(self.agents[item.agent], item.function)(
                            *item.args, **item.keyword_args
                        )
            except Exception:
                logger.exception("Error handling queue item")
                # Raise error to avoid unexpectedly continuing execution in the restricted code with a None result
                raise
            return None
