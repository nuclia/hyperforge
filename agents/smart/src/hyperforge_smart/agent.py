import asyncio
import json
import logging
from dataclasses import dataclass, field
from time import time
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple
from uuid import uuid4

from hyperforge.agent import Agent
from hyperforge.configure import agent
from hyperforge.context.agent import ContextAgent, build_context_agent
from hyperforge.definition import FunctionDefinition
from hyperforge.interaction import Feedback
from hyperforge.manager import Manager
from hyperforge.memory.memory import QuestionMemory
from hyperforge.models import Chunk, Context, TrackingInfo
from hyperforge.result_payload import budget_from_config, inspect_text_blocks
from hyperforge.utils import iterate_tools_resp
from nuclia.lib.nua_responses import (
    Author,
    ChatModel,
    Message,
    Tool,
    UserPrompt,
)
from pydantic import BaseModel, ConfigDict, Field
from sentry_sdk import capture_exception

from hyperforge_smart.config import SmartAgentConfig
from hyperforge_smart.prompts import (
    PLAN_EXECUTE_EXECUTOR_SYSTEM_PROMPT_TEMPLATE,
    PLAN_EXECUTE_PLANNER_JSON_SCHEMA,
    PLAN_EXECUTE_PLANNER_PROMPT_TEMPLATE,
    PLAN_EXECUTE_PLANNER_SYSTEM_PROMPT,
    REACTIVE_SYSTEM_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)

TOOL_NAME_SEPARATOR = "__"


@dataclass
class ToolError:
    """Represents a tool execution error, kept out of the final context.

    Tracks which tool call (name + arguments) caused the error so the
    LLM can be informed and decide on an alternative approach.
    """

    tool_name: str
    tool_arguments: Dict[str, Any]
    error: str

    def __str__(self) -> str:
        return self.error


@dataclass
class SkippedToolCall:
    """Represents a tool call that was intentionally skipped by the orchestrator."""

    tool_name: str
    tool_arguments: Dict[str, Any]
    reason: str

    def __str__(self) -> str:
        return self.reason


ToolOutcome = Literal["useful", "empty", "error"]


@dataclass
class ToolAttempt:
    tool_name: str
    tool_arguments: Dict[str, Any]
    outcome: ToolOutcome
    detail: str


TASK_COMPLETE_TOOL = Tool(
    name="task_complete",
    description="Call this tool when you have gathered enough information to answer the question and no more tools are needed.",
    parameters={
        "type": "object",
        "properties": {},
    },
)

USER_FEEDBACK_TOOL = Tool(
    name="user_feedback",
    description="Ask the user a clarifying question when you need more information to proceed.",
    parameters={
        "type": "object",
        "properties": {
            "decision_id": {
                "type": "string",
                "description": "Stable identifier for the underlying decision. Reuse it if the question is rephrased.",
            },
            "question": {
                "type": "string",
                "description": "The question to ask the user",
            },
        },
        "required": ["decision_id", "question"],
    },
)


@dataclass
class PlanIteration:
    """Record of a single planner iteration and its execution results."""

    plan_steps: List[Dict[str, Any]] = field(default_factory=list)
    plan_summary: str = ""
    results: List[Tuple[str, Any]] = field(default_factory=list)
    results_summary: str = ""
    tool_attempts: List[ToolAttempt] = field(default_factory=list)


class RegisteredAgent(BaseModel):
    """A registered context agent with optional description and schema for the planner."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent: "ContextAgent" = Field(
        ...,
        title="Agent",
        description="The context agent",
    )
    description: Optional[str] = Field(
        None,
        title="Description",
        description="Description of what this agent does, used by the planner",
    )
    available_functions: Optional[Dict[str, FunctionDefinition]] = Field(
        None,
        title="Exposed functions",
        description="List of functions exposed by this agent",
    )


@agent(
    id="smart",
    agent_type="context",
    title="Smart Agent",
    description="Use multiple agents in a smart way to gather context to answer questions.",
    config_schema=SmartAgentConfig,
)
class SmartAgent(Agent[SmartAgentConfig], ContextAgent):
    __published_functions__: ClassVar[Dict[str, FunctionDefinition]] = {
        "smart_planner": FunctionDefinition(
            name="smart_planner",
            description="Execute multiple agents in a smart way to gather context to answer questions.",
            parameters={
                "question": {
                    "type": "string",
                    "description": "The question to answer by gathering context from registered agents.",
                },
            },
        )
    }
    config: SmartAgentConfig
    registered_agents: List[RegisteredAgent]

    @staticmethod
    def _normalize_tool_arguments(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: SmartAgent._normalize_tool_arguments(value[key])
                for key in sorted(value)
            }
        if isinstance(value, list):
            return [SmartAgent._normalize_tool_arguments(item) for item in value]
        return value

    @classmethod
    def _tool_call_key(cls, tool_name: str, tool_arguments: Dict[str, Any]) -> str:
        normalized_arguments = cls._normalize_tool_arguments(tool_arguments or {})
        return json.dumps(
            {"tool_name": tool_name, "tool_arguments": normalized_arguments},
            sort_keys=True,
            ensure_ascii=True,
            default=str,
        )

    @staticmethod
    def _context_has_useful_content(context: Context) -> bool:
        return any(
            [
                bool(context.chunks),
                bool(context.structured),
                bool(context.json_objects),
                bool(context.prompts),
                bool(context.images),
                bool(context.summary.strip()),
                bool(context.image_urls),
            ]
        )

    @classmethod
    def _classify_tool_result(cls, result: Any) -> ToolOutcome:
        if isinstance(result, (ToolError, SkippedToolCall)):
            return "error"
        if result is None:
            return "empty"
        if isinstance(result, Context):
            return "useful" if cls._context_has_useful_content(result) else "empty"
        if isinstance(result, list):
            if not result:
                return "empty"
            if all(isinstance(item, Context) for item in result):
                return (
                    "useful"
                    if any(cls._context_has_useful_content(item) for item in result)
                    else "empty"
                )
            return "useful"
        if isinstance(result, str):
            return "useful" if result.strip() else "empty"
        if isinstance(result, (dict, tuple, set)):
            return "useful" if len(result) > 0 else "empty"
        return "useful"

    @staticmethod
    def _planner_attempts_from_registry(
        attempted_tool_calls: Dict[str, ToolAttempt],
    ) -> List[ToolAttempt]:
        return list(attempted_tool_calls.values())

    async def inner_from_config(
        self, config: SmartAgentConfig, agent_id: Optional[str] = None
    ):
        # Build registered agents - convert the agent config to actual agent instances
        registered_agents_list = []
        registered_agents_exposed_functions = (
            config.registered_agents_exposed_functions or {}
        )
        registered_agents_descriptions = config.registered_agents_descriptions or {}
        for reg_agent_config in config.registered_agents or []:
            # Build the actual agent from its config
            agent_instance = await build_context_agent(reg_agent_config)

            if agent_instance is not None:
                # Create RegisteredAgent with the built instance
                available_functions = None
                agent_id = agent_instance.context_config.id
                if agent_id is None:
                    agent_id = agent_instance.agent_id
                exposed_functions = registered_agents_exposed_functions.get(
                    agent_id, None
                )
                description = registered_agents_descriptions.get(agent_id, "")
                if isinstance(exposed_functions, list) and exposed_functions:
                    available_functions = {
                        function_id: function_definition
                        for function_id, function_definition in agent_instance.__published_functions__.items()
                        if function_id in exposed_functions
                    }
                elif exposed_functions is None or (
                    isinstance(exposed_functions, list) and not exposed_functions
                ):
                    available_functions = agent_instance.__published_functions__
                registered_agents_list.append(
                    RegisteredAgent(
                        agent=agent_instance,
                        description=description,
                        available_functions=available_functions,
                    )
                )
        # FALLBACK AND NEXT AGENT HIDDEN IN THIS ITERATION
        # fallback_agent = await build_context_agent(config.get("fallback"))
        # next_agent = await build_context_agent(config.get("next_agent"))

        await self.context_from_config(config)
        self.registered_agents = registered_agents_list

    def get_agent_by_id(self, agent_id: str) -> Optional[RegisteredAgent]:
        """Find a registered agent by its ID."""
        for reg_agent in self.registered_agents:
            if reg_agent.agent.agent_id == agent_id:
                return reg_agent
        return None

    def build_tools(self) -> List[Tool]:
        """Convert all registered agent functions into Tool objects for native tool calling."""
        tools: List[Tool] = []
        for reg_agent in self.registered_agents:
            if not reg_agent.available_functions:
                continue
            agent_id = reg_agent.agent.agent_id or ""
            for function_id, func_def in reg_agent.available_functions.items():
                tool_name = f"{function_id}{TOOL_NAME_SEPARATOR}{agent_id}"
                description = func_def.description
                if reg_agent.description:
                    description = f"{reg_agent.description} — {description}"
                tools.append(
                    Tool(
                        name=tool_name,
                        description=description,
                        parameters={
                            "type": "object",
                            "properties": func_def.parameters,
                            "additionalProperties": False,
                        },
                    )
                )
        tools.append(TASK_COMPLETE_TOOL)
        if self.config.enable_user_feedback:
            tools.append(USER_FEEDBACK_TOOL)
        return tools

    def build_tools_description(self) -> str:
        """Return a human-readable text block describing all available tools for the planner prompt."""
        lines: List[str] = []
        for reg_agent in self.registered_agents:
            if not reg_agent.available_functions:
                continue
            agent_id = reg_agent.agent.agent_id or ""
            for function_id, func_def in reg_agent.available_functions.items():
                tool_name = f"{function_id}{TOOL_NAME_SEPARATOR}{agent_id}"
                description = func_def.description
                if reg_agent.description:
                    description = f"{reg_agent.description} — {description}"
                param_lines = []
                for param_name, param_info in (func_def.parameters or {}).items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    param_lines.append(
                        f"    - {param_name} ({param_type}): {param_desc}"
                    )
                params_text = (
                    "\n".join(param_lines) if param_lines else "    (no parameters)"
                )
                lines.append(f"- **{tool_name}**: {description}\n{params_text}")
        return "\n\n".join(lines) if lines else "(no tools available)"

    def _reactive_tools(
        self,
        tools: List[Tool],
        attempted_tool_calls: Dict[str, ToolAttempt],
    ) -> List[Tool]:
        """Hide parameterless tools after they returned no result in this run."""
        available_tools = []
        for tool in tools:
            properties = tool.parameters.get("properties", {})
            empty_call_key = self._tool_call_key(tool.name, {})
            empty_attempt = attempted_tool_calls.get(empty_call_key)
            if (
                not properties
                and empty_attempt is not None
                and empty_attempt.outcome == "empty"
            ):
                continue
            available_tools.append(tool)
        return available_tools

    def _process_results(
        self,
        results: List[Tuple[str, Any]],
        collected_contexts: Optional[List[Context]] = None,
    ) -> List[str]:
        """Process tool results and optionally retain each returned context.

        ToolError results are included in the text summaries (so the LLM
        is aware of the failure) but are never stored in the context.
        """
        result_texts: List[str] = []
        for action_info, result in results:
            overflow = self._inspect_result(result)
            if isinstance(result, ToolError):
                error_text = overflow.render() if overflow is not None else result.error
                result_texts.append(f"[{action_info}]:\n{error_text}")
                continue

            if isinstance(result, SkippedToolCall):
                result_texts.append(f"[{action_info}]:\n{result.reason}")
                continue

            if isinstance(result, Context):
                contexts = [result]
            elif isinstance(result, list) and all(
                isinstance(item, Context) for item in result
            ):
                contexts = result
            else:
                contexts = []

            if overflow is not None:
                error_text = overflow.render()
                result_texts.append(f"[{action_info}]:\n{error_text}")
                if collected_contexts is not None:
                    collected_contexts.append(
                        self._synthetic_result_context(action_info, error_text)
                    )
                continue

            if collected_contexts is not None:
                if contexts:
                    for ctx in contexts:
                        for chunk in ctx.chunks:
                            chunk.action = action_info
                        collected_contexts.append(ctx)
                else:
                    collected_contexts.append(
                        self._synthetic_result_context(action_info, str(result))
                    )

            for ctx in contexts:
                if ctx.summary:
                    result_texts.append(f"[{action_info}]:\n{ctx.summary}")
                else:
                    result_texts.append(f"[{action_info}]:\n{ctx.context_markdown()}")
            if not contexts:
                result_texts.append(f"[{action_info}]:\n{result}")

        return result_texts

    def _synthetic_result_context(self, action_info: str, text: str) -> Context:
        return Context(
            agent_id=self.config.id or "smart_agent",
            original_question_uuid=None,
            actual_question_uuid=None,
            question="",
            source="smart_agent",
            agent="smart_agent",
            title=action_info,
            chunks=[
                Chunk(
                    chunk_id=uuid4().hex,
                    text=text,
                    action=action_info,
                    origin_agent=self.config.module,
                )
            ],
        )

    def _inspect_result(self, result: Any):
        if isinstance(result, ToolError):
            texts = [result.error]
        else:
            contexts = (
                [result]
                if isinstance(result, Context)
                else result
                if isinstance(result, list)
                and all(isinstance(item, Context) for item in result)
                else []
            )
            texts = []
            for context in contexts:
                if context.summary.strip():
                    texts.append(context.summary)
                else:
                    texts.extend(chunk.render() for chunk in context.chunks)
                    texts.extend(context.structured)
            if not texts and contexts:
                texts = [""]
            if not contexts:
                texts = [str(result)]
        return inspect_text_blocks(texts, budget_from_config(self.config))

    async def choose_tools(
        self,
        manager: Manager,
        messages: List[Message],
        tools: List[Tool],
        system_override: Optional[str] = None,
        tracking: TrackingInfo | None = None,
    ) -> Tuple[Any, float, float]:
        """Call the LLM with available tools and return its tool selections."""

        system = system_override or REACTIVE_SYSTEM_PROMPT_TEMPLATE.render(
            extra_instructions=self.config.extra_prompt or ""
        )
        model = self.config.executor_model

        item = ChatModel(
            question="",
            user_id=f"smart_planner-{self.config.module}",
            generative_model=model,
            tools=tools,
            user_prompt=UserPrompt(
                prompt=f"{system}\n\nChoose the best tool or tools for the task. Call task_complete when you have enough information."
            ),
            format_prompt=False,
            system=system,
            chat_history=messages,
        )
        resp, input_tokens, output_tokens = await manager.execute_raw(
            item, tracking=tracking
        )
        return resp, input_tokens, output_tokens

    async def _report_tool_error(
        self,
        memory: QuestionMemory,
        title: str,
        error: str,
        tool_name: str,
        tool_arguments: Dict[str, Any],
    ) -> Tuple[str, ToolError]:
        """Log a tool error to memory and return a ToolError result."""
        overflow = inspect_text_blocks([error], budget_from_config(self.config))
        if overflow is not None:
            error = overflow.render()
        await memory.add_step(
            step_module=self.config.module,
            step_title=self.step_title(
                "Tool result rejected" if overflow is not None else title
            ),
            step_reason=error,
            step_agent_path=f"/context/{self.config.id or 'default'}",
            step_value=overflow.trace_value() if overflow is not None else "Error",
            timeit=0,
        )
        return tool_name, ToolError(
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            error=error,
        )

    async def execute_tool_call(
        self,
        memory: QuestionMemory,
        manager: Manager,
        tool_name: str,
        tool_arguments: Dict[str, Any],
    ) -> Tuple[str, Any]:
        """Parse a tool name, look up the agent/function, and execute it."""
        parts = tool_name.split(TOOL_NAME_SEPARATOR, 1)
        if len(parts) != 2:
            return await self._report_tool_error(
                memory,
                "Invalid tool name",
                f"Invalid tool name format: {tool_name!r}",
                tool_name,
                tool_arguments,
            )

        function_id, agent_id = parts
        reg_agent = self.get_agent_by_id(agent_id)

        if reg_agent is None:
            return await self._report_tool_error(
                memory,
                "Agent not found",
                f"Agent {agent_id!r} not found in registered agents",
                tool_name,
                tool_arguments,
            )

        if (
            not reg_agent.available_functions
            or function_id not in reg_agent.available_functions
        ):
            return await self._report_tool_error(
                memory,
                "Function not found",
                f"Function {function_id!r} not found in agent {agent_id!r}",
                tool_name,
                tool_arguments,
            )

        action_attr = getattr(reg_agent.agent, function_id, None)
        if action_attr is None:
            return await self._report_tool_error(
                memory,
                "Function not found",
                f"Function {function_id!r} not found in agent {agent_id!r} instance",
                tool_name,
                tool_arguments,
            )

        try:
            result = await action_attr(
                memory=memory,
                manager=manager,
                **tool_arguments,
            )
        except TypeError as e:
            return await self._report_tool_error(
                memory,
                "LLM Execution error",
                f"Binding error executing tool {function_id!r} of agent {agent_id!r}: {e}",
                tool_name,
                tool_arguments,
            )
        except Exception as e:
            logger.exception(
                f"Error executing tool {function_id!r} of agent {agent_id!r}"
            )
            capture_exception(e)
            return await self._report_tool_error(
                memory,
                "LLM Execution error",
                f"Error executing tool {function_id!r} of agent {agent_id!r}: {e}",
                tool_name,
                tool_arguments,
            )

        action_info = f"{function_id} of {agent_id}"
        if tool_arguments:
            action_info += f" with parameters {tool_arguments}"
        overflow = self._inspect_result(result)
        if overflow is not None:
            await memory.add_step(
                step_module=self.config.module,
                step_title=self.step_title("Tool result rejected"),
                step_reason=f"{action_info} exceeded the configured LLM context safety budget.",
                step_agent_path=f"/context/{self.config.id or 'default'}",
                step_value=overflow.trace_value(),
                timeit=0,
            )
        return action_info, result

    async def _preload_registered_agents(
        self, manager: Manager, memory: QuestionMemory
    ) -> None:
        """Call ``preload`` on every registered agent that supports it.

        Agents whose ``__published_functions__`` dict is empty after
        ``inner_from_config`` (e.g. ``MCPAgent``) declare their tools at
        runtime via this hook.  After preloading, ``available_functions`` on
        the ``RegisteredAgent`` wrapper is refreshed so ``build_tools`` picks
        up the newly discovered functions.

        Agents that do not override ``preload`` (the base no-op) are unaffected.
        """
        for reg_agent in self.registered_agents:
            await reg_agent.agent.preload(manager, memory)
            if (
                not reg_agent.available_functions
                and reg_agent.agent.__published_functions__
            ):
                reg_agent.available_functions = reg_agent.agent.__published_functions__

    async def smart_planner(
        self,
        question: str,
        memory: QuestionMemory,
        manager: Manager,
        question_uuid: Optional[str] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> List[Context]:
        """Entry point: dispatches to the appropriate reasoning mode."""
        if question_uuid is None:
            question_uuid = uuid4().hex

        await self._preload_registered_agents(manager, memory)

        session_context_parts: List[str] = []
        reactive_history_messages: List[Message] = []

        if self.config.history:
            if self.config.planning_mode == "reactive":
                reactive_history_messages = await memory.get_chat_history()
                interactions = sum(
                    1
                    for message in reactive_history_messages
                    if message.author == Author.USER
                )
            else:
                qa_history, interactions = await memory.context_history()
                session_context_parts.append(
                    f"## Previous questions and answers in this session:\n{qa_history}"
                )
            await memory.add_step(
                step_module=self.config.module,
                step_title=self.step_title("History check"),
                step_value="Included {} interactions of Q&A history".format(
                    interactions
                ),
                step_reason="",
                timeit=0,
                step_agent_path=f"/context/{self.config.id if self.config.id else 'default'}",
                input_nuclia_tokens=0.0,
                output_nuclia_tokens=0.0,
            )

        session_context = "\n\n".join(session_context_parts)

        if self.config.planning_mode == "plan_execute":
            return await self._plan_and_execute(
                question=question,
                memory=memory,
                manager=manager,
                question_uuid=question_uuid,
                extra_context=extra_context,
                session_context=session_context,
            )
        return await self._reactive_loop(
            question=question,
            memory=memory,
            manager=manager,
            question_uuid=question_uuid,
            extra_context=extra_context,
            session_context=session_context,
            history_messages=reactive_history_messages,
        )

    async def _execute_tool_calls_turn(
        self,
        memory: QuestionMemory,
        manager: Manager,
        messages: List[Message],
        tool_calls: List[Tuple[str, Any]],
        turn_label: str,
        resolved_feedback: Optional[Dict[str, str]] = None,
        collected_contexts: Optional[List[Context]] = None,
        attempted_tool_calls: Optional[Dict[str, ToolAttempt]] = None,
    ) -> List[Tuple[str, Any]]:
        """Handle one turn of tool calls.

        If the LLM requested user feedback, sends the feedback request, records it as a
        step, stores it via _process_results and returns (results, True) so the caller
        can ``continue`` to the next iteration without executing other tools.

        Otherwise executes all tool calls in parallel, records an execution step and
        returns (results, False).
        """
        agent_path = f"/context/{self.config.id or 'default'}"

        # --- user_feedback path ---
        if any(name == "user_feedback" for name, _ in tool_calls):
            for name, args in tool_calls:
                if name == "user_feedback":
                    decision_id: Optional[str] = (
                        args.get("decision_id") if args else None
                    )
                    feedback_question: Optional[str] = (
                        args.get("question") if args else None
                    )
                    if feedback_question:
                        decision_id = decision_id or (
                            f"legacy:{feedback_question.strip().lower()}"
                        )
                        if (
                            resolved_feedback is not None
                            and decision_id in resolved_feedback
                        ):
                            cached_feedback_result = (
                                "user_feedback",
                                "Feedback resolved. Do not request this feedback again. "
                                f"Decision ID: {decision_id}\n"
                                f"Question: {feedback_question}\n"
                                f"Response: {resolved_feedback[decision_id]}",
                            )
                            result_texts = self._process_results(
                                [cached_feedback_result],
                                collected_contexts=collected_contexts,
                            )
                            if result_texts:
                                messages.append(
                                    Message(
                                        author=Author.NUCLIA,
                                        text="\n\n".join(result_texts),
                                    )
                                )
                            return [cached_feedback_result]
                        feedback = Feedback(
                            request_id=memory.get_session_id(),
                            question=feedback_question,
                            module=self.config.module,
                            agent_id=self.config.id or "default",
                            data=None,
                            timeout_ms=self.config.feedback_timeout,
                            response_schema={
                                "type": "object",
                                "properties": {"response": {"type": "string"}},
                                "required": ["response"],
                            },
                        )
                        answer = await memory.send_feedback(feedback)
                        feedback_text = (
                            answer.response
                            if answer is not None
                            else "(No response received)"
                        )
                        messages.append(Message(author=Author.USER, text=feedback_text))
                        logger.info(f"Received user feedback response: {feedback_text}")
                        await memory.add_step(
                            step_module=self.config.module,
                            step_title=self.step_title(f"User feedback {turn_label}"),
                            step_reason="User feedback requested and received.",
                            step_agent_path=agent_path,
                            step_value=f"Feedback question: {feedback_question}\nFeedback response: {feedback_text}",
                            timeit=0,
                        )
                        feedback_result: Tuple[str, Any] = (
                            "user_feedback",
                            "Feedback resolved. Do not request this feedback again. "
                            f"Decision ID: {decision_id}\n"
                            f"Question: {feedback_question}\n"
                            f"Response: {feedback_text}",
                        )
                        if resolved_feedback is not None:
                            resolved_feedback[decision_id] = feedback_text
                        result_texts = self._process_results(
                            [feedback_result], collected_contexts=collected_contexts
                        )
                        if result_texts:
                            messages.append(
                                Message(
                                    author=Author.NUCLIA,
                                    text="\n\n".join(result_texts),
                                )
                            )
                        return [feedback_result]
            return []

        # --- normal tool execution path ---
        attempted_tool_calls = (
            attempted_tool_calls if attempted_tool_calls is not None else {}
        )
        skipped_results: List[Tuple[str, Any]] = []
        calls_to_execute: List[Tuple[str, Dict[str, Any]]] = []

        for name, args in tool_calls:
            if name == "task_complete":
                continue
            tool_arguments = args or {}
            tool_key = self._tool_call_key(name, tool_arguments)
            previous_attempt = attempted_tool_calls.get(tool_key)
            if previous_attempt is not None and previous_attempt.outcome in {
                "empty",
                "error",
            }:
                skipped_results.append(
                    (
                        name,
                        SkippedToolCall(
                            tool_name=name,
                            tool_arguments=tool_arguments,
                            reason=(
                                "Skipped repeated tool call because the same tool "
                                "with the same arguments already returned an "
                                f"{previous_attempt.outcome} result in this run."
                            ),
                        ),
                    )
                )
                continue

            attempted_tool_calls[tool_key] = ToolAttempt(
                tool_name=name,
                tool_arguments=tool_arguments,
                outcome="useful",
                detail="pending",
            )
            calls_to_execute.append((name, tool_arguments))

        async def execute_with_delegation(
            tool_name: str, tool_arguments: Dict[str, Any]
        ) -> Tuple[str, Any]:
            await memory.add_step(
                step_module=self.config.module,
                step_title=self.step_title(f"Delegating to {tool_name}"),
                step_reason="Starting selected tool",
                step_agent_path=agent_path,
                step_value=tool_name,
                timeit=0,
            )
            return await self.execute_tool_call(
                memory, manager, tool_name, tool_arguments
            )

        executed_results = []
        if calls_to_execute and self.config.parallel_tool_calls:
            executed_results = await asyncio.gather(
                *[
                    execute_with_delegation(name, args)
                    for name, args in calls_to_execute
                ]
            )
        elif calls_to_execute:
            for name, args in calls_to_execute:
                executed_results.append(await execute_with_delegation(name, args))

        results = [*executed_results, *skipped_results]
        for (tool_name, tool_arguments), (_, result) in zip(
            calls_to_execute, executed_results
        ):
            tool_key = self._tool_call_key(tool_name, tool_arguments)
            attempted_tool_calls[tool_key] = ToolAttempt(
                tool_name=tool_name,
                tool_arguments=tool_arguments,
                outcome=self._classify_tool_result(result),
                detail=str(result),
            )
        for _, skipped_result in skipped_results:
            tool_key = self._tool_call_key(
                skipped_result.tool_name, skipped_result.tool_arguments
            )
            previous_attempt = attempted_tool_calls.get(tool_key)
            if previous_attempt is not None:
                previous_attempt.detail = str(skipped_result)

        result_texts = self._process_results(
            list(results), collected_contexts=collected_contexts
        )
        result_summary = "; ".join(
            f"{info}: {'context' if isinstance(res, Context) else type(res).__name__}"
            for info, res in results
        )
        await memory.add_step(
            step_module=self.config.module,
            step_title=self.step_title(f"Execution {turn_label}"),
            step_reason=f"Executed {len(results)} tool(s) and collected results",
            step_agent_path=agent_path,
            step_value=f"Results: {result_summary}",
            timeit=0,
        )
        if result_texts:
            messages.append(
                Message(author=Author.NUCLIA, text="\n\n".join(result_texts))
            )
        return list(results)

    async def _reactive_loop(
        self,
        question: str,
        memory: QuestionMemory,
        manager: Manager,
        question_uuid: str,
        extra_context: Optional[Dict[str, Any]] = None,
        session_context: str = "",
        history_messages: Optional[List[Message]] = None,
    ) -> List[Context]:
        t0 = time()

        tools = self.build_tools()
        messages: List[Message] = list(history_messages or [])
        if session_context:
            messages.append(
                Message(
                    author=Author.NUCLIA,
                    text=(
                        "The following context from the current session may be relevant to answer the user's question, or can be used to rephrase the question or guide the model:\n\n"
                        + session_context
                    ),
                )
            )
        messages.append(Message(author=Author.USER, text=question))

        contexts: List[Context] = []

        iteration = 0
        finished = False
        total_input_tokens = 0.0
        total_output_tokens = 0.0
        resolved_feedback: Dict[str, str] = {}
        attempted_tool_calls: Dict[str, ToolAttempt] = {}
        while not finished and iteration < self.config.max_iterations:
            iteration += 1
            available_tools = self._reactive_tools(tools, attempted_tool_calls)

            resp, input_tokens, output_tokens = await self.choose_tools(
                manager,
                messages,
                available_tools,
                tracking=memory.get_tracking_info(),
            )
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            tool_calls = list(iterate_tools_resp(resp))

            tool_names = [name for name, _ in tool_calls]
            tool_detail = (
                ", ".join(
                    f"{name}({', '.join(f'{k}={v!r}' for k, v in (args or {}).items())})"
                    for name, args in tool_calls
                )
                or "none"
            )

            await memory.add_step(
                step_module=self.config.module,
                step_title=self.step_title(
                    f"Reactive iteration {iteration}/{self.config.max_iterations}"
                ),
                step_reason=f"LLM selected {len(tool_calls)} tool(s): {', '.join(tool_names) or 'none'}",
                step_agent_path=f"/context/{self.config.id or 'default'}",
                step_value=f"Tool calls: {tool_detail}",
                timeit=0,
                input_nuclia_tokens=input_tokens,
                output_nuclia_tokens=output_tokens,
            )

            if not tool_calls:
                finished = True
                break

            # Execute tool calls (handles user_feedback and normal tool calls)
            _ = await self._execute_tool_calls_turn(
                memory=memory,
                manager=manager,
                messages=messages,
                tool_calls=tool_calls,
                turn_label=f"iteration {iteration}/{self.config.max_iterations}",
                resolved_feedback=resolved_feedback,
                collected_contexts=contexts,
                attempted_tool_calls=attempted_tool_calls,
            )
            if any(name == "task_complete" for name, _ in tool_calls):
                finished = True

        reason = (
            "Task complete signal received"
            if finished
            else f"Reached max iterations ({self.config.max_iterations})"
        )
        await memory.add_step(
            step_module=self.config.module,
            step_title=self.step_title("Reactive mode completed"),
            step_reason=reason,
            step_agent_path=f"/context/{self.config.id or 'default'}",
            step_value=f"Completed after {iteration} iteration(s). Total tokens: {total_input_tokens} in / {total_output_tokens} out",
            timeit=time() - t0,
            input_nuclia_tokens=total_input_tokens,
            output_nuclia_tokens=total_output_tokens,
        )

        return contexts

    async def _call_planner(
        self,
        manager: Manager,
        question: str,
        history: List[PlanIteration],
        tools_description: str,
        session_context: str = "",
        tracking: TrackingInfo | None = None,
    ) -> Tuple[Dict[str, Any], float, float]:
        """Call the planner LLM to produce a structured retrieval plan."""
        prompt = PLAN_EXECUTE_PLANNER_PROMPT_TEMPLATE.render(
            question=question,
            tools_description=tools_description,
            history=history,
            extra_instructions=self.config.extra_prompt or "",
            session_context=session_context,
        )
        full_prompt = PLAN_EXECUTE_PLANNER_SYSTEM_PROMPT + "\n\n" + prompt

        # Commented until we fix the reasoning issue around json output
        # if self.config.planner_reasoning:
        #     item = ChatModel(
        #         user_id=f"smart_planner_plan-{self.config.module}",
        #         question="",
        #         user_prompt=UserPrompt(prompt=full_prompt),
        #         generative_model=self.config.planner_model,
        #         format_prompt=False,
        #         json_schema=PLAN_EXECUTE_PLANNER_JSON_SCHEMA,
        #         system=PLAN_EXECUTE_PLANNER_SYSTEM_PROMPT,
        #         citations=False,
        #         reasoning=Reasoning(effort="medium"),
        #         max_tokens=20_000,
        #     )
        #     resp, input_tokens, output_tokens = await manager.execute_raw(item)
        #     if resp.object is None:
        #         raise Exception("No object from planner")
        #     return resp.object, input_tokens, output_tokens

        response, input_tokens, output_tokens = await manager.execute_json(
            user_id=f"smart_planner_plan-{self.config.module}",
            prompt=full_prompt,
            schema=PLAN_EXECUTE_PLANNER_JSON_SCHEMA,
            model=self.config.planner_model,
            system=PLAN_EXECUTE_PLANNER_SYSTEM_PROMPT,
            tracking=tracking,
        )
        return response, input_tokens, output_tokens

    async def _call_executor(
        self,
        memory: QuestionMemory,
        manager: Manager,
        question: str,
        steps: List[Dict[str, Any]],
        summary: str,
        tools: List[Tool],
        messages: List[Message],
        resolved_feedback: Dict[str, str],
    ) -> Tuple[List[Tuple[str, Any]], float, float, List[ToolAttempt]]:
        """Run the executor LLM turn: call tools guided by the current plan steps."""
        system = PLAN_EXECUTE_EXECUTOR_SYSTEM_PROMPT_TEMPLATE.render(
            question=question,
            steps=steps,
            summary=summary,
            extra_instructions=self.config.extra_prompt or "",
        )
        all_results: List[Tuple[str, Any]] = []
        total_input_tokens = 0.0
        total_output_tokens = 0.0
        finished = False
        max_executor_turns = self.config.max_iterations
        attempted_tool_calls: Dict[str, ToolAttempt] = {}

        executor_turn = 0
        while not finished and executor_turn < max_executor_turns:
            executor_turn += 1
            resp, input_tokens, output_tokens = await self.choose_tools(
                manager=manager,
                messages=messages,
                tools=tools,
                system_override=system,
                tracking=memory.get_tracking_info(),
            )
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            tool_calls = list(iterate_tools_resp(resp))

            if not tool_calls:
                finished = True
                break

            executable_calls = [
                (name, args) for name, args in tool_calls if name != "task_complete"
            ]
            if not executable_calls:
                finished = True
                break

            results = await self._execute_tool_calls_turn(
                memory=memory,
                manager=manager,
                messages=messages,
                tool_calls=executable_calls,
                turn_label=f"executor turn {executor_turn}/{max_executor_turns}",
                resolved_feedback=resolved_feedback,
                attempted_tool_calls=attempted_tool_calls,
            )
            all_results.extend(results)
            if any(name == "task_complete" for name, _ in tool_calls):
                finished = True

        return (
            all_results,
            total_input_tokens,
            total_output_tokens,
            self._planner_attempts_from_registry(attempted_tool_calls),
        )

    async def _plan_and_execute(
        self,
        question: str,
        memory: QuestionMemory,
        manager: Manager,
        question_uuid: str,
        extra_context: Optional[Dict[str, Any]] = None,
        session_context: str = "",
    ) -> List[Context]:
        """Plan-execute reasoning mode: planner drafts a plan, executor runs tools, repeat."""
        t0 = time()
        agent_path = f"/context/{self.config.id or 'default'}"

        contexts: List[Context] = []

        history: List[PlanIteration] = []
        iteration = 0
        total_input_tokens = 0.0
        total_output_tokens = 0.0
        executor_messages = [Message(author=Author.USER, text=question)]
        resolved_feedback: Dict[str, str] = {}
        status = "done"
        steps: List[Dict[str, Any]] = []

        # Build tools once for the entire plan-execute cycle
        tools = self.build_tools()
        tools_description = self.build_tools_description()

        while iteration < self.config.max_iterations:
            iteration += 1

            # PLANNER
            plan_response, plan_in_tokens, plan_out_tokens = await self._call_planner(
                manager=manager,
                question=question,
                history=history,
                tools_description=tools_description,
                session_context=session_context,
                tracking=memory.get_tracking_info(),
            )
            total_input_tokens += plan_in_tokens
            total_output_tokens += plan_out_tokens

            status = plan_response.get("status", "done")
            reasoning = plan_response.get("reasoning", "")
            summary = plan_response.get("summary", "")
            steps = plan_response.get("steps", [])

            if steps:
                plan_summary = f"{len(steps)} step(s): " + "; ".join(
                    s.get("description", "?") for s in steps
                )
            else:
                plan_summary = "(no steps)"

            step_detail = "\n".join(
                f"  {i + 1}. [{s.get('reason', '')}] {s.get('description', '?')}"
                for i, s in enumerate(steps)
            )
            await memory.add_step(
                step_module=self.config.module,
                step_title=self.step_title(
                    f"Planner iteration {iteration}/{self.config.max_iterations}"
                ),
                step_reason=f"Status: {status}. Reasoning: {reasoning}",
                step_agent_path=agent_path,
                step_value=f"Plan: {plan_summary}\nSummary so far: {summary or '(initial)'}\nSteps:\n{step_detail}",
                timeit=0,
                input_nuclia_tokens=plan_in_tokens,
                output_nuclia_tokens=plan_out_tokens,
            )

            if status == "done" or not steps:
                break

            # EXECUTOR
            current_iteration = PlanIteration(
                plan_steps=steps,
                plan_summary=plan_summary,
            )

            (
                iteration_results,
                exec_in_tokens,
                exec_out_tokens,
                iteration_attempts,
            ) = await self._call_executor(
                memory=memory,
                manager=manager,
                question=question,
                steps=steps,
                summary=summary,
                tools=tools,
                messages=executor_messages,
                resolved_feedback=resolved_feedback,
            )
            total_input_tokens += exec_in_tokens
            total_output_tokens += exec_out_tokens

            result_texts = self._process_results(
                iteration_results, collected_contexts=contexts
            )
            results_summary = (
                "\n\n".join(result_texts) if result_texts else "(no results)"
            )
            for action_info, result in iteration_results:
                overflow = self._inspect_result(result)
                current_iteration.results.append(
                    (action_info, overflow.render() if overflow is not None else result)
                )
            current_iteration.results_summary = results_summary
            current_iteration.tool_attempts = iteration_attempts
            history.append(current_iteration)

            exec_result_summary = (
                "; ".join(
                    f"{info}: {'context' if isinstance(res, Context) else type(res).__name__}"
                    for info, res in iteration_results
                )
                or "(no results)"
            )

            await memory.add_step(
                step_module=self.config.module,
                step_title=self.step_title(
                    f"Executor completed iteration {iteration}/{self.config.max_iterations}"
                ),
                step_reason=f"Executed plan with {len(steps)} step(s). Summary: {summary}",
                step_agent_path=agent_path,
                step_value=f"Executed {len(iteration_results)} tool call(s). Results: {exec_result_summary}",
                timeit=0,
                input_nuclia_tokens=exec_in_tokens,
                output_nuclia_tokens=exec_out_tokens,
            )

        done_reason = (
            "Planner signaled done"
            if (status == "done" or not steps)
            else f"Reached max iterations ({self.config.max_iterations})"
        )
        await memory.add_step(
            step_module=self.config.module,
            step_title=self.step_title("Plan-execute mode completed"),
            step_reason=done_reason,
            step_agent_path=agent_path,
            step_value=f"Completed after {iteration} planning iteration(s). Total tokens: {total_input_tokens} in / {total_output_tokens} out",
            timeit=time() - t0,
            input_nuclia_tokens=total_input_tokens,
            output_nuclia_tokens=total_output_tokens,
        )

        return contexts

    async def _get_question_context(
        self,
        memory: QuestionMemory,
        manager: Manager,
        question_uuid: str,
        question: str,
        flow_id: str,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, str]]:
        contexts = await self.smart_planner(
            memory=memory,
            manager=manager,
            question_uuid=question_uuid,
            question=question,
            extra_context=extra_context,
        )

        missing = await self.save_contexts_and_return_missing(
            contexts=contexts,
            question=question,
            memory=memory,
            manager=manager,
            flow_id=flow_id,
        )
        return [missing] if missing is not None else []
