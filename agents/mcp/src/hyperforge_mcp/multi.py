from time import time
from typing import Any, Dict, List, Optional, Tuple

from hyperforge.agent import Agent
from hyperforge.context.agent import ContextAgent
from hyperforge.interaction import Feedback, PromptFeedbackSchema
from hyperforge.manager import Manager
from hyperforge.memory import QuestionMemory
from hyperforge.models import Context, TrackingInfo
from hyperforge.trace import trace_agent
from hyperforge.utils import iterate_tools_resp
from mcp import types
from nuclia.lib.nua_responses import Image, Message

from hyperforge import logger
from hyperforge_mcp.agent import EXIT_LOOP_TOOLS, MCPAgent
from hyperforge_mcp.config import MCPAgentConfig, MultiMCPAgentConfig
from hyperforge_mcp.tools import (
    PROMPT_CHOOSE_TEMPLATE,
    SYSTEM_SUMMARIZE_TOOLS,
    TOOLS_CHOOSE_TEMPLATE,
    TOOLS_SUMMARIZE_EXAMPLES_TEMPLATE,
)


class MultiMCPAgent(Agent[MultiMCPAgentConfig], ContextAgent):
    config: MultiMCPAgentConfig  # type: ignore
    agents: list[MCPAgent]
    main_agent: MCPAgent

    def __init__(
        self, config: MultiMCPAgentConfig, agent_id: Optional[str] = None
    ) -> None:
        super().__init__(config, agent_id)
        self.agents = []

    async def inner_from_config(
        self, config: MultiMCPAgentConfig, agent_id: Optional[str] = None
    ):
        configs = config.configs

        agents: list[MCPAgent] = []
        for mcp_config in configs:
            if mcp_config.module != "mcp":
                raise Exception("All configs must be of module mcp")
            agent = await MCPAgent.from_config(mcp_config, agent_id=agent_id)
            agents.append(agent)

        await self.context_from_config(config)
        self.agents = agents
        self.main_agent = MCPAgent(config=MCPAgentConfig(module="mcp", id=agent_id))  # type: ignore

    async def summarize_tools(
        self,
        manager: Manager,
        tools: list[types.Tool],
        tracking: TrackingInfo | None = None,
    ) -> Tuple[str, float, float]:
        if not tools:
            return "No tools available.", 0.0, 0.0
        # Summarize tools
        system = SYSTEM_SUMMARIZE_TOOLS
        prompt = TOOLS_SUMMARIZE_EXAMPLES_TEMPLATE.render(
            mcp_id=self.config.id, tools=tools
        )
        response = await manager.execute(
            prompt=prompt,
            model=self.config.summarize_model,
            system=system,
            user_id="multi_summarize_tools",
            tracking=tracking,
        )
        return response[:3]

    async def summarize_prompts(
        self,
        manager: Manager,
        prompts: list[types.Prompt],
        tracking: TrackingInfo | None = None,
    ) -> Tuple[str, float, float]:
        if not prompts:
            return "No prompts available.", 0.0, 0.0
        # Summarize prompts
        prompt = "The following are the prompts available:\n"
        system = None
        for p in prompts:
            prompt += f"- {p.name}: {p.description}\n"
        response = await manager.execute(
            prompt=prompt,
            model=self.config.summarize_model,
            system=system,
            user_id="multi_summarize_tools",
            tracking=tracking,
        )
        return response[:3]

    async def multi_choose_tool(
        self,
        manager: Manager,
        memory: QuestionMemory,
        messages: list[Message],
        images: list[Image],
    ):
        t0 = time()
        prompt_feedback = TOOLS_CHOOSE_TEMPLATE.render(tools=self.main_agent.tools)
        resp, input_tokens, output_tokens = await manager.execute_json(
            model=self.config.tool_choice_model,
            prompt=prompt_feedback,
            user_id="mcp_no_feedback",
            schema={
                "type": "object",
                "properties": {
                    "tool_id": {
                        "type": "string",
                        "description": "id of the tool to use",
                    },
                },
            },
            tracking=memory.get_tracking_info(),
        )
        tool_id: str = resp["tool_id"]

        await memory.add_step(
            step_module=self.config.module,
            step_title=self.step_title("Tool selection"),
            step_reason="",
            step_agent_path=f"/context/{self.config.id if self.config.id else 'default'}",
            step_value="mcp_no_feedback",
            timeit=time() - t0,
            input_nuclia_tokens=input_tokens,
            output_nuclia_tokens=output_tokens,
        )
        agent_obj = next(
            (agent for agent in self.agents if agent.config.id == tool_id), None
        )
        if agent_obj is None:
            raise Exception(f"No tool found with id {tool_id}")
        return await agent_obj.choose_tool(
            manager=manager, messages=messages, images=images
        )

    async def get_multi_tool_selection_prompt(
        self, manager: Manager, question: str, memory: QuestionMemory, context: Context
    ) -> Tuple[list[Message], list[Image]]:
        # Choose prompt subset
        if self.main_agent.session is None:
            raise Exception("MCP session not initialized")

        t0 = time()
        agent_obj = None
        messages: List[Message] = []
        images: List[Image] = []

        if self.config.interaction and len(self.main_agent.prompts) > 0:
            feedback = Feedback(
                request_id=memory.get_session_id(),
                question="Choose proper use case",
                module=self.config.module,
                agent_id=self.config.id or "default",
                data=self.main_agent.prompts,
                timeout_ms=self.config.feedback_timeout,
                response_schema=PromptFeedbackSchema.model_json_schema(),
            )
            answer = await memory.send_feedback(feedback)
            if answer is not None and answer.request_id == memory.get_session_id():
                prompt_feedback = PromptFeedbackSchema.model_validate_json(
                    answer.response
                )  # Validate JSON
                agent_obj = next(
                    (
                        agent
                        for agent in self.agents
                        if agent.config.id == prompt_feedback.prompt_id
                    ),
                    None,
                )

        if agent_obj is None:
            # We will use LLM to choose the prompt
            prompt_feedback_str = PROMPT_CHOOSE_TEMPLATE.render(
                prompts=self.main_agent.prompts
            )
            resp, input_tokens, output_tokens = await manager.execute_json(
                model=self.config.tool_choice_model,
                prompt=prompt_feedback_str,
                user_id="mcp_no_feedback",
                schema={
                    "type": "object",
                    "properties": {
                        "prompt_id": {
                            "type": "string",
                            "description": "id of the prompt to use",
                        },
                    },
                },
                tracking=memory.get_tracking_info(),
            )
            prompt_id: str = resp["prompt_id"]

            await memory.add_step(
                step_module=self.config.module,
                step_title=self.step_title("Prompt selection"),
                step_reason="",
                step_agent_path=f"/context/{self.config.id if self.config.id else 'default'}",
                step_value="mcp_no_feedback",
                timeit=time() - t0,
                input_nuclia_tokens=input_tokens,
                output_nuclia_tokens=output_tokens,
            )
            agent_obj = next(
                (agent for agent in self.agents if agent.config.id == prompt_id), None
            )

        if agent_obj is not None:
            messages, images = await agent_obj.get_tool_selection_prompt(
                manager, question, memory, context
            )

        return messages, images

    async def mcp_interaction(
        self, memory: QuestionMemory, manager: Manager, question: str, context: Context
    ) -> Tuple[float, float]:
        """
        Interact with the MCP server to get the context for the question.
        This method will use the MCP server to call tools and get structured data.
        """

        total_input_tokens = 0.0
        total_output_tokens = 0.0
        messages, images = await self.get_multi_tool_selection_prompt(
            manager, question, memory, context
        )
        resp, input_tokens, output_tokens = await self.multi_choose_tool(
            manager=manager, images=images, messages=messages, memory=memory
        )
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        for tool_name, tool_arguments in iterate_tools_resp(resp):
            await self.main_agent.process_tool(
                memory, tool_name, tool_arguments, context, messages, images
            )

        if self.config.work_chain is False:
            logger.debug("Exiting loop on tool")
            return input_tokens, output_tokens

        count = 0
        finished = False
        while count > self.config.max_turns is False and finished is False:
            count += 1
            resp, input_tokens, output_tokens = await self.main_agent.choose_tool(
                manager, images, messages, EXIT_LOOP_TOOLS
            )
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            for tool_name, tool_arguments in iterate_tools_resp(resp):
                if tool_name == "task_complete":
                    logger.debug("Exiting loop on task_complete tool")
                    finished = True
                    break
                await self.main_agent.process_tool(
                    memory, tool_name, tool_arguments, context, messages, images
                )
        return total_input_tokens, total_output_tokens

    @trace_agent
    async def get_question_context(
        self,
        memory: QuestionMemory,
        manager: Manager,
        question_uuid: str,
        question: str,
        flow_id: str,
        extra_context: Optional[Dict[str, Any]] = None,
    ):
        global_input_tokens = 0.0
        global_output_tokens = 0.0
        for agent in self.agents:
            if agent.config.id is None:
                continue
            try:
                await agent.initialize(manager, memory)
            except KeyError:
                raise Exception("No MCP driver found")
            response, input_tokens, output_tokens = await self.summarize_tools(
                manager,
                agent.tools,
                tracking=memory.get_tracking_info(),
            )
            global_input_tokens += input_tokens
            global_output_tokens += output_tokens
            self.main_agent.tools.append(
                types.Tool(name=agent.config.id, description=response, inputSchema={})
            )  # type: ignore

            response, input_tokens, output_tokens = await self.summarize_prompts(
                manager,
                agent.prompts,
                tracking=memory.get_tracking_info(),
            )
            global_input_tokens += input_tokens
            global_output_tokens += output_tokens
            self.main_agent.prompts.append(
                types.Prompt(name=agent.config.id, description=response)
            )  # type: ignore

        context = Context(
            agent_id=self.agent_id,
            original_question_uuid=memory.original_question_uuid,
            actual_question_uuid=question_uuid,
            question=question,
            structured=[],
            source="multi_mcp",
            agent="mcp",
        )

        t0 = time()

        (
            input_tokens,
            output_tokens,
        ) = await self.mcp_interaction(memory, manager, question, context)

        await memory.add_step(
            step_module=self.config.module,
            step_title=self.step_title("Interaction finished"),
            step_reason="",
            step_agent_path=f"/context/{self.config.id if self.config.id else 'default'}",
            step_value="mcp_multi_interaction_finished",
            timeit=time() - t0,
            input_nuclia_tokens=input_tokens,
            output_nuclia_tokens=output_tokens,
        )

        missing = await self.save_ctx_and_return_missing(
            context=context,
            question=question,
            memory=memory,
            manager=manager,
            flow_id=flow_id,
        )
        if missing is not None and self.fallback is not None:
            missing_uuid, missing_question = missing
            await self.fallback.get_question_context(
                memory,
                manager,
                question_uuid=missing_uuid,
                question=missing_question,
                flow_id=flow_id,
            )
