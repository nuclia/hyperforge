import string
from functools import lru_cache
from time import time
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple, cast
from uuid import uuid4

from hyperforge import logger
from hyperforge.configure import get_agent_klass
from hyperforge.context.config import ContextAgentConfig
from hyperforge.definition import FunctionDefinition
from hyperforge.manager import Manager, ModelParam
from hyperforge.memory import Context, QuestionMemory
from hyperforge.prompts import (
    NEXT_REPHRASE_JSON_SCHEMA,
    NEXT_REPHRASE_PROMPT_SYSTEM,
    NEXT_REPHRASE_PROMPT_TEMPLATE,
    VALIDATE_JSON_SCHEMA,
    VALIDATE_MULTIPLE_CONTEXTS_JSON_SCHEMA,
    VALIDATE_MULTIPLE_CONTEXTS_PROMPT_TEMPLATE,
    VALIDATE_OR_ANSWER_PROMPT_TEMPLATE,
)
from hyperforge.trace import trace_agent


@lru_cache(maxsize=64)
def generate_ctx_block_id(n: int) -> str:
    """Generates block ids in the form block-AA, block-AB, ..., block-AZ, block-BA, ..., block-ZZ"""
    # XXX: We use letters to identify block since uuids confuse the LLMs and with numbers it confuses the block numbers with footnote numbers
    if n < 0:
        raise ValueError("Number must be non-negative")
    if n >= 26 * 26:
        logger.warning(
            "Block ID exceeds maximum limit for citations,", extra={"block_id": n}
        )
        n = n % (26 * 26)
    letters = string.ascii_uppercase
    first = letters[(n // 26)]
    second = letters[n % 26]
    return "block-" + first + second


async def build_context_agent(config: ContextAgentConfig) -> "ContextAgent":
    agent_class = get_agent_klass(config.module)
    assert issubclass(agent_class, ContextAgent), (
        f"Agent {config.module} is not a ContextAgent"
    )
    return await agent_class.from_config(config)  # type: ignore[return-type]  # ty: ignore[invalid-return-type]


class ContextAgent:
    fallback: Optional["ContextAgent"] = None
    next_agent: Optional["ContextAgent"] = None
    agent_description: str = "Agent that provides context to answer questions."
    __published_functions__: ClassVar[Dict[str, FunctionDefinition]] = {}
    exposed_functions: Optional[List[str]] = None
    agent_id: str

    async def preload(self, manager: "Manager", memory: "QuestionMemory") -> None:
        """Lifecycle hook called by SmartAgent before tool discovery.

        Override in subclasses that need a live connection (e.g. MCP) to
        populate ``__published_functions__`` at runtime.  The default
        implementation is a no-op.
        """
        pass

    @property
    def context_config(self) -> ContextAgentConfig:
        return cast(ContextAgentConfig, self.config)  # type: ignore

    async def inner_from_config(self, config: Any, agent_id: Optional[str] = None):
        await self.context_from_config(config)

    async def context_from_config(self, config: ContextAgentConfig):
        # Shared logic for fallback and next_agent
        if config.fallback:
            self.fallback = await build_context_agent(config.fallback)
        if config.next_agent:
            self.next_agent = await build_context_agent(config.next_agent)

    async def validate_contexts_and_answer(
        self,
        memory: QuestionMemory,
        manager: Manager,
        contexts: list[Context],
        question: str,
        images: bool = False,
        fast_answer: bool = False,
        user_id: str = "rao_answer_summary",
        use_stored_context_prompts: bool = False,
        title: Optional[str] = None,
    ) -> Tuple[Literal["yes", "no", "error"], Optional[str], Optional[str]]:
        """Validate several contexts and update each relevant context independently."""
        t0 = time()
        model = self.context_config.context_validation_model
        module = self.context_config.module
        ident = self.context_config.id if self.context_config.id else "default"
        title = title or self.context_config.title or module
        context_by_id = {context.id: context for context in contexts}
        block_targets: dict[str, tuple[str, str]] = {}
        prompt_contexts = []
        validation_images = {}
        block_index = 0

        for context_index, context in enumerate(contexts):
            context.citations = None
            blocks = {}
            for chunk in context.chunks:
                block_id = generate_ctx_block_id(block_index)
                block_index += 1
                blocks[block_id] = chunk.render()
                block_targets[block_id] = (context.id, chunk.chunk_id)
            for structured_index, structured in enumerate(context.structured):
                block_id = generate_ctx_block_id(block_index)
                block_index += 1
                blocks[block_id] = structured
                block_targets[block_id] = (
                    context.id,
                    f"structured-{structured_index}",
                )
            prompt_contexts.append(
                {
                    "context_id": context.id,
                    "title": context.title,
                    "summary": context.summary,
                    "blocks": blocks,
                }
            )
            validation_images.update(
                {
                    f"context-{context_index}-{image_id}": image
                    for image_id, image in context.images.items()
                }
            )

        prompt = VALIDATE_MULTIPLE_CONTEXTS_PROMPT_TEMPLATE.render(
            question=question,
            contexts=prompt_contexts,
        )
        try:
            response, input_tokens, output_tokens = await manager.execute_json(
                user_id=user_id + f"-{module}",
                images=validation_images if images else {},
                prompt=prompt,
                schema=VALIDATE_MULTIPLE_CONTEXTS_JSON_SCHEMA,
                model=model,
            )
        except Exception as e:
            error_message = f"Error executing validation of contexts for {module} with question '{question}': {e}"
            await memory.add_step(
                step_module=module,
                step_title=f"{title}: Context summary error",
                step_reason="Error in summary generation",
                step_value="Error in summary generation",
                timeit=time() - t0,
                step_agent_path=f"/context/{ident}",
                error=error_message,
            )
            return "error", question, error_message

        relevant_contexts = 0
        for context_result in response.get("contexts", []):
            ctx = context_by_id.get(context_result.get("context_id"))
            if ctx is None:
                continue
            cited_items = []
            for block_id in dict.fromkeys(context_result.get("citations", [])):
                target = block_targets.get(block_id)
                if target is not None and target[0] == ctx.id:
                    cited_items.append(target[1])

            answer = context_result.get("answer", "")
            if not answer and not cited_items:
                continue

            relevant_contexts += 1
            ctx.citations = cited_items
            if answer and "not enough data to answer this" not in answer.lower():
                ctx.summary = answer
            else:
                ctx.summary = ""

        missing = response.get("missing_info_query") or None
        await memory.add_step(
            step_module=module,
            step_title=f"{title}: Context validation",
            step_reason=f"Selected {relevant_contexts} of {len(contexts)} contexts",
            step_value="No missing information" if not missing else missing,
            timeit=time() - t0,
            input_nuclia_tokens=input_tokens,
            output_nuclia_tokens=output_tokens,
            step_agent_path=f"/context/{ident}",
        )
        return "yes" if relevant_contexts else "no", missing, None

    async def validate_ctx_and_answer(
        self,
        memory: QuestionMemory,
        manager: Manager,
        context: Context,
        question: str,
        images: bool = False,
        fast_answer: bool = False,
        user_id: str = "rao_answer_summary",
        use_stored_context_prompts: bool = False,
        title: Optional[str] = None,
    ) -> Tuple[Literal["yes", "no", "error"], Optional[str], Optional[str]]:
        """
        Validate if the context is useful to answer the question and attempt use it to answer the user's question.

        Reports back if the context is useful, what is missing, any error during summarization.

        It will also save the answer attempt and the citations in the context object.
        """
        t0 = time()
        model = self.context_config.context_validation_model
        module = self.context_config.module
        ident = self.context_config.id if self.context_config.id else "default"

        citations = None
        contexts = [x.render() for x in context.chunks]
        contexts.extend([x for x in context.structured])

        contexts_items = {generate_ctx_block_id(i): x for i, x in enumerate(contexts)}
        contexts_items.update(
            {
                generate_ctx_block_id(i + len(context.chunks)): x
                for i, x in enumerate(context.structured)
            }
        )
        block_id_to_chunk = {
            generate_ctx_block_id(i): chunk.chunk_id
            for i, chunk in enumerate(context.chunks)
        }
        block_id_to_chunk.update(
            {
                generate_ctx_block_id(i + len(context.chunks)): f"structured-{i}"
                for i in range(len(context.structured))
            }
        )

        extra_prompts = [prompt.render() for prompt in context.prompts]
        prompt = VALIDATE_OR_ANSWER_PROMPT_TEMPLATE.render(
            question=question,
            contexts=contexts_items,
            extra_prompts=extra_prompts if use_stored_context_prompts else "",
        )
        try:
            (
                response,
                input_tokens,
                output_tokens,
            ) = await manager.execute_json(
                user_id=user_id + f"-{module}",
                images=context.images if images else {},
                prompt=prompt,
                schema=VALIDATE_JSON_SCHEMA,
                model=model,
            )
            answer = response.get("answer", "")
            missing = response.get("missing_info_query", None)
            useful = response.get("useful", "yes")
            reason = response.get("reason", "")
            citations = response.get("citations", None)
        except Exception as e:
            error_message = f"Error executing validation of contexts for {module} with question '{question}': {e}"
            await memory.add_step(
                step_module=module,
                step_title=f"{title}: Context summary error",
                step_reason="Error in summary generation",
                step_value="Error in summary generation",
                timeit=time() - t0,
                step_agent_path=f"/context/{ident}",
                error=error_message,
            )
            return "error", question, error_message

        # removing this for now
        # if useful == "yes" and fast_answer:
        #     await memory.add_answer(answer, module="ask", agent_path=f"/context/{ident}")

        if citations is not None:
            cited_chunks = [
                block_id_to_chunk[citation]
                for citation in set(citations)
                if citation in block_id_to_chunk
            ]
            context.citations = cited_chunks

        context.summary = (
            answer if "not enough data to answer this" not in answer.lower() else ""
        )

        await memory.add_step(
            step_module=module,
            step_title=f"{title}: Context validation",
            step_reason=reason,
            step_value="No missing information" if not missing else missing,
            timeit=time() - t0,
            input_nuclia_tokens=input_tokens,
            output_nuclia_tokens=output_tokens,
            step_agent_path=f"/context/{ident}",
        )

        if missing is not None and missing.strip() != "":
            return useful, missing, None
        return useful, None, None

    async def rephrase(
        self,
        memory: QuestionMemory,
        manager: Manager,
        contexts: Dict[str, Any],
        ident: str,
        question: str,
        question_uuid: str,
        model: ModelParam = "",
        module: str = "agent",
        user_id: str = "next",
        title: Optional[str] = None,
    ) -> Tuple[str, str]:
        t0 = time()
        rephrased = False
        prompt = NEXT_REPHRASE_PROMPT_TEMPLATE.render(
            question=question,
            contexts=list(contexts.values()),
            info=self.agent_description,
            extra_info=self.context_config.context_aware_rephrasing_prompt,
        )
        try:
            (
                response,
                input_tokens,
                output_tokens,
            ) = await manager.execute_json(
                user_id=user_id + f"-{module}",
                system=NEXT_REPHRASE_PROMPT_SYSTEM,
                prompt=prompt,
                schema=NEXT_REPHRASE_JSON_SCHEMA,
                model=model,
            )
            rephrased_question = response.get("rephrased_question", "")
            needed = response.get("needed", True)
            reason = response.get("reason", "")
        except Exception as e:
            error_message = f"Error executing rephrase when executing next agent {module} with question '{question}': {e}"
            await memory.add_step(
                step_module=module,
                step_title=f"{title}: Rephrase error",
                step_reason="Error in rephrase generation",
                step_value="Error in rephrase generation",
                timeit=time() - t0,
                step_agent_path=f"/context/{ident}",
                error=error_message,
            )
            return question, question_uuid
        if (
            rephrased_question.strip() != ""
            and needed is True
            and "not enough data to answer this" not in rephrased_question.lower()
        ):
            rephrased_question_uuid = uuid4().hex
            rephrased = True

        await memory.add_step(
            step_module=module,
            step_title=f"{title}: Rephrase",
            step_reason=reason,
            step_value="No need for rephrasing"
            if rephrased is False
            else rephrased_question,
            timeit=time() - t0,
            input_nuclia_tokens=input_tokens,
            output_nuclia_tokens=output_tokens,
            step_agent_path=f"/context/{ident}",
        )
        return (
            rephrased_question if rephrased else question,
            rephrased_question_uuid if rephrased else question_uuid,
        )

    async def _get_question_context(
        self,
        memory: QuestionMemory,
        manager: Manager,
        question_uuid: str,
        question: str,
        flow_id: str,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[str, str]]:
        """To be implemented by child classes to get context for the question.
        Should return a list of (question_uuid, question) tuples representing missing questions.
        """
        raise NotImplementedError()

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
        if extra_context is not None:
            question, question_uuid = await self.rephrase(
                memory=memory,
                manager=manager,
                question_uuid=question_uuid,
                question=question,
                contexts=extra_context,
                model=self.context_config.rephrase_model,
                module=self.context_config.module,
                user_id="next_rephrase",
                ident=self.agent_id,
            )

        # Missing questions should be a list of (question_uuid, question) tuples
        missing_questions = await self._get_question_context(
            memory, manager, question_uuid, question, flow_id=flow_id
        )

        if self.fallback is not None:
            for missing_uuid, missing in missing_questions:
                await self.fallback.get_question_context(
                    memory,
                    manager,
                    question_uuid=missing_uuid,
                    question=missing,
                    flow_id=flow_id,
                )

        if self.next_agent is not None:
            extra_context = extra_context or {}
            for agent in [self, self.fallback]:
                if agent:
                    answer_summaries = memory.get_agent_answer_summaries(
                        flow_id=flow_id, agent_id=agent.agent_id
                    )
                    if answer_summaries:
                        extra_context[agent.agent_id] = "\n".join(answer_summaries)

            await self.next_agent.get_question_context(
                memory,
                manager,
                question_uuid,
                question,
                extra_context=extra_context,
                flow_id=flow_id,
            )

    async def save_ctx_and_return_missing(
        self,
        *,
        memory: QuestionMemory,
        manager: Manager,
        question: str,
        context: Context,
        flow_id: str,
        use_stored_context_prompts: bool = False,
    ) -> tuple[str, str] | None:
        # We trigger context validation and answer attempt only if we have a fallback or next agent or pruning is enabled
        if (
            self.fallback is not None
            or self.next_agent is not None
            or self.context_config.prune_context
            or use_stored_context_prompts
        ) and self.context_config.context_validation_model:
            (
                useful,
                missing,
                validate_error,
            ) = await self.validate_ctx_and_answer(
                memory,
                manager,
                context,
                question=question,
                use_stored_context_prompts=use_stored_context_prompts,
            )

            if useful == "yes" or validate_error is not None:
                if self.context_config.prune_context and validate_error is None:
                    context.prune_to_citations()
                await memory.save_context(flow_id=flow_id, context=context)
            missing_question = (
                None
                if missing is None or missing.strip() == ""
                else (uuid4().hex, missing)
            )
            return missing_question
        else:
            await memory.save_context(flow_id=flow_id, context=context)
        return None

    async def save_contexts_and_return_missing(
        self,
        *,
        memory: QuestionMemory,
        manager: Manager,
        question: str,
        contexts: list[Context],
        flow_id: str,
    ) -> tuple[str, str] | None:
        """Validate, prune, and save several contexts as one retrieval result."""
        should_validate = (
            self.fallback is not None
            or self.next_agent is not None
            or self.context_config.prune_context
        ) and self.context_config.context_validation_model

        if should_validate:
            useful, missing, validate_error = await self.validate_contexts_and_answer(
                memory,
                manager,
                contexts,
                question=question,
                images=any(context.images for context in contexts),
            )
            if useful == "yes" or validate_error is not None:
                for context in contexts:
                    if self.context_config.prune_context and validate_error is None:
                        if context.citations is None:
                            continue
                        if context.citations == []:
                            if not context.summary:
                                continue
                            context.chunks = []
                            context.structured = []
                        else:
                            context.prune_to_citations()
                    elif validate_error is None and context.citations is None:
                        continue
                    await memory.save_context(
                        flow_id=flow_id, context=context, agent_id=self.agent_id
                    )
            return (
                None
                if missing is None or missing.strip() == ""
                else (uuid4().hex, missing)
            )

        for context in contexts:
            await memory.save_context(
                flow_id=flow_id, context=context, agent_id=self.agent_id
            )
        return None
