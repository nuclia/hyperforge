from time import time
from typing import List, Optional, overload

from hyperforge import PROMPT_ENVIRONMENT
from hyperforge.agent import Agent
from hyperforge.configure import agent
from hyperforge.context.agent import generate_ctx_block_id
from hyperforge.manager import Manager
from hyperforge.memory import QuestionMemory
from hyperforge.models import AnswerCitations, CitationMetadata, Context
from hyperforge.trace import trace_agent
from nuclia.lib.nua_responses import ChatModel, Tool, ToolChoiceAuto, UserPrompt
from nuclia_models.predict.generative_responses import ToolCall

from hyperforge_summarize.config import SummarizeAgentConfig
from hyperforge_summarize.prompts import MARKDOWN_TWO_LEVELS_CITATIONS_PROMPT_ADJUSTMENT

DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. Your role is to provide accurate, clear, and well-structured answers based strictly on the information provided to you.
Key principles:
- Answer only using the information in the provided context
- Do not use external knowledge, assumptions, or prior experience
- Maintain a professional and informative tone
- Be concise yet thorough
- If information is insufficient, acknowledge this clearly

Always follow any additional instructions provided about format, style, or domain-specific behavior."""

SUMMARIZE_PROMPT_CONVERSATIONAL = """
{% if rules -%}
# Generation Rules
{% for rule in rules -%}
- {{ rule }}
{% endfor -%}
{% endif -%}


## Question
{{ question }}

## Provided Context
[START OF CONTEXT]
{{ context }}
[END OF CONTEXT]

## Answering Guidelines
- Carefully read all context; it may be lengthy or detailed
- Do not omit or overlook any relevant information
- If the context is incomplete or insufficient, try to provide a partial answer and encourage the user to clarify their question
- Read carefully any extra instructions below if provided and use them to answer

{% if prompt -%}
## Additional Instructions for answering
{{ prompt }}
{% endif -%}

{% if chat_history -%}
## Previous conversation history
- {{ chat_history }}
{% endif -%}

{% if extra_prompts -%}
## Extra Prompts to consider for generating the answer
This information was used to generate the context.
Use it to help generate a better answer and follow any specific instructions it may contain about the format or style of the answer.
{% for extra in extra_prompts -%}
- {{ extra }}
{% endfor -%}
{% endif -%}

Now provide your answer to the question: {{ question }}
"""

SUMMARIZE_PROMPT = """
{% if rules -%}
# Generation Rules
{% for rule in rules -%}
- {{ rule }}
{% endfor -%}
{% endif -%}

## Question
{{ question }}

## Provided Context
[START OF CONTEXT]
{{ context }}
[END OF CONTEXT]

## Answering Guidelines
- Carefully read all context; it may be lengthy or detailed
- Do not omit or overlook any relevant information
- If the context is incomplete or insufficient, state: "Not enough data to answer this."
- Read carefully any extra instructions below if provided and use them to answer

{% if prompt -%}
## Additional Instructions for answering
- {{ prompt }}
{% endif -%}

{% if chat_history -%}
## Previous conversation history
- {{ chat_history }}
{% endif -%}


{% if extra_prompts -%}
## Extra Prompts to consider for generating the answer
This information was used to generate the context.
Use it to help generate a better answer and follow any specific instructions it may contain about the format or style of the answer.
{% for extra in extra_prompts -%}
- {{ extra }}
{% endfor -%}
{% endif -%}

Now provide your answer to the question: {{ question }}
"""

SUMMARIZE_PROMPT_TEMPLATE = PROMPT_ENVIRONMENT.from_string(SUMMARIZE_PROMPT)
SUMMARIZE_PROMPT_CONVERSATIONAL_TEMPLATE = PROMPT_ENVIRONMENT.from_string(
    SUMMARIZE_PROMPT_CONVERSATIONAL
)


@agent(
    id="summarize",
    agent_type="generation",
    title="Summarize",
    description="Summarize the provided context.",
    config_schema=SummarizeAgentConfig,
)
class SummarizeAgent(Agent[SummarizeAgentConfig]):
    __root_agent__ = True
    config: SummarizeAgentConfig

    @overload
    async def __call__(
        self,
        memory: QuestionMemory,
        manager: Manager,
        tools: None = None,
    ) -> None: ...

    @overload
    async def __call__(
        self,
        memory: QuestionMemory,
        manager: Manager,
        tools: list[Tool],
    ) -> None | dict[str, list[ToolCall]]: ...

    @trace_agent
    async def __call__(
        self,
        memory: QuestionMemory,
        manager: Manager,
        tools: list[Tool] | None = None,
    ) -> None | dict[str, list[ToolCall]]:
        citations_enabled = self.config.citations

        # For each context in memory add context query, summary and answer onto a text and the initial question
        # In cases when the original question has been rephrased, use the rephrased question
        questions = memory.get_questions()
        if len(questions) == 1:
            question = questions[0][1]
        else:
            question = memory.original_question

        session_context_parts: List[str] = []

        if self.config.history:
            qa_history, interactions = await memory.context_history()
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
            session_context_parts.append(
                f"## Previous questions and answers in this session:\n{qa_history}"
            )

        session_context = "\n\n".join(session_context_parts)

        PROMPT_TEMPLATE = (
            SUMMARIZE_PROMPT_CONVERSATIONAL_TEMPLATE
            if self.config.conversational
            else SUMMARIZE_PROMPT_TEMPLATE
        )
        prompt = self.config.prompt
        extra_prompts: List[str] = []
        if self.config.include_mcp_prompts:
            extra_prompts = memory.get_prompt_texts()

        if citations_enabled:
            # Add citation ids to each context so they can be referenced in the answer
            for index, context in enumerate(memory.contexts):
                context.citations_id = generate_ctx_block_id(index)

        prompt = PROMPT_TEMPLATE.render(
            question=question,
            context=memory.contexts_markdown()
            if citations_enabled and self.config.force_chunk_level_citations
            else memory.contexts_minimal(),
            prompt=prompt,
            extra_prompts=extra_prompts,
            rules=self.config.rules,
            chat_history=session_context,
        )

        if citations_enabled:
            # Adjust the prompt so that the model returns citations
            prompt += MARKDOWN_TWO_LEVELS_CITATIONS_PROMPT_ADJUSTMENT

        t0 = time()
        images = {}
        for memory_context in memory.contexts:
            if memory_context.images:
                images.update(memory_context.images)
        chat_model = ChatModel(
            user_id="summarize",
            question="",
            user_prompt=UserPrompt(prompt=prompt),
            system=self.config.system_prompt
            if self.config.system_prompt
            else DEFAULT_SYSTEM_PROMPT,
            format_prompt=False,
            generative_model=self.config.model,
            query_context_images=images,
            max_tokens=5000,
            chat_history=await memory.get_chat_history(),
            tools=tools if tools else [],
            tool_choice=ToolChoiceAuto(),
        )

        agent_path = f"/generation/{self.config.id if self.config.id else 'default'}"
        # Pass memory so execute_raw streams automatically when memory.streaming is True.
        # Streaming is skipped when tools are active (tool calls are not streamable).
        streaming_memory = memory if not tools else None
        resp, input_tokens, output_tokens = await manager.execute_raw(
            chat_model,
            memory=streaming_memory,
            module="summarize",
            agent_path=agent_path,
            tracking=memory.get_tracking_info(),
        )
        answer = resp.answer
        end_code = resp.code

        # Fall back to original question and full contexts
        if end_code == "-2":  # indicates not enough data
            prompt = PROMPT_TEMPLATE.render(
                question=memory.original_question,
                context=memory.contexts_markdown(),
                prompt=prompt,
                extra_prompts=extra_prompts,
                rules=self.config.rules,
            )
            if citations_enabled:
                # Adjust the prompt so that the model returns citations
                prompt += MARKDOWN_TWO_LEVELS_CITATIONS_PROMPT_ADJUSTMENT

            chat_model.user_prompt = UserPrompt(prompt=prompt)
            resp, input_tokens, output_tokens = await manager.execute_raw(
                chat_model,
                memory=streaming_memory,
                module="summarize",
                agent_path=agent_path,
                tracking=memory.get_tracking_info(),
            )
            answer = resp.answer
            end_code = resp.code

        if not resp.tools or answer:
            # Only add answer if not a tool call or if answer is present (maybe some models return both)
            await memory.add_answer(
                answer,
                module="summarize",
                agent_path=f"/generation/{self.config.id if self.config.id else 'default'}",
                citations=build_answer_citations(answer, memory.contexts)
                if citations_enabled
                else None,
            )
        memory.is_answered = end_code != "-2"
        await memory.add_step(
            step_module=self.config.module,
            step_title=self.step_title("Summarize"),
            step_value=str(answer),
            step_reason="Summarized",
            step_agent_path=f"/generation/{self.config.id if self.config.id else 'default'}",
            timeit=time() - t0,
            input_nuclia_tokens=input_tokens,
            output_nuclia_tokens=output_tokens,
        )
        return resp.tools


def build_answer_citations(answer: str, contexts: list[Context]) -> AnswerCitations:
    result = AnswerCitations()
    # Build a map of citation_id to context
    citation_map = {
        context.citations_id: context
        for context in contexts
        if context.citations_id is not None
    }

    # Parse citations in the answer
    citations_in_answer: set[str] = set()
    for line in answer.splitlines():
        if line.startswith("[") and "]: block-" in line:
            citation_id = line.split("]: ")[1].strip()
            citations_in_answer.add(citation_id)

    for citation_id in citations_in_answer:
        try:
            context_citation_id, chunk_index = _parse_citation_id(citation_id)
        except ValueError:
            # Unknown format, skip
            continue

        if context_citation_id not in citation_map:
            # Unknown citation, skip
            continue

        context: Context = citation_map[context_citation_id]
        origin_urls: list[str] = []

        if chunk_index is not None:
            # This is a citation to a specific chunk of a context
            try:
                chunk = context.chunks[chunk_index]
                if chunk.origin_url:
                    origin_urls.append(chunk.origin_url)
            except IndexError:
                # Chunk index is out of range, skip
                pass

        else:
            # This is a citation to a summarized context
            for chunk in context.chunks:
                if (
                    not context.citations or chunk.chunk_id in context.citations
                ) and chunk.origin_url:
                    origin_urls.append(chunk.origin_url)

            if not origin_urls:
                for chunk in context.chunks:
                    if chunk.origin_url:
                        origin_urls.append(chunk.origin_url)

        if origin_urls:
            origin_urls = list(dict.fromkeys(origin_urls))

        result.metadata[citation_id] = CitationMetadata(
            context_id=context.id,
            origin_urls=origin_urls,
            chunk_index=chunk_index,
        )

    return result


def _parse_citation_id(citation_id: str) -> tuple[str, Optional[int]]:
    """Parse a citation id into context citation id and optional chunk index.

    Examples:
        - "block-abc123" -> ("block-abc123", None)
        - "block-abc123-0" -> ("block-abc123", 0)
    """
    if citation_id.count("-") >= 2:
        # Assume the last part is the chunk index
        parts = citation_id.rsplit("-", 1)
        context_citation_id = parts[0]
        try:
            chunk_index = int(parts[1])
            return context_citation_id, chunk_index
        except ValueError:
            raise ValueError("Invalid citation id format")
    else:
        return citation_id, None
