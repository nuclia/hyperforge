"""
Passthrough generation agent.

Returns the retrieved context chunks directly as the answer, with no LLM call.
Useful for testing and for workflows where the context agent already provides
a complete answer (e.g. the built-in ``static`` context agent).
"""

from hyperforge.agent import Agent
from hyperforge.configure import agent
from hyperforge.manager import Manager
from hyperforge.memory import QuestionMemory
from hyperforge.trace import trace_agent

from agents.passthrough.src.hyperforge_passthrough.config import PassthroughAgentConfig


@agent(
    id="passthrough",
    agent_type="generation",
    title="Passthrough",
    description=(
        "Return the retrieved context directly as the answer without any LLM call. "
        "Useful for testing or when the context agent already contains the final answer."
    ),
    config_schema=PassthroughAgentConfig,
)
class PassthroughAgent(Agent[PassthroughAgentConfig]):
    __root_agent__ = True

    @trace_agent
    async def __call__(
        self,
        memory: QuestionMemory,
        manager: Manager,
    ) -> None:
        if self.config.rich_context:
            # Gather all context data and pass it through add_answer so it
            # arrives in possible_answer with all fields populated.
            # convert_arag_answer_to_content will parse it thoroughly.
            chunks = [
                chunk for ctx in memory.contexts for chunk in ctx.chunks if chunk.text
            ]
            structured_data = [
                s for ctx in memory.contexts for s in ctx.structured if s
            ]
            images = {
                k: img
                for ctx in memory.contexts
                for k, img in (ctx.images or {}).items()
            }
            image_urls = [
                url for ctx in memory.contexts for url in ctx.image_urls if url
            ]
            await memory.add_answer(
                "",
                "passthrough",
                "/generation/passthrough",
                chunks=chunks,
                structured=structured_data,
                images=images,
                image_urls=image_urls,
            )
            return

        # Default behaviour: concatenate all context chunks as a plain-text answer.
        parts = []
        for ctx in memory.contexts:
            for chunk in ctx.chunks:
                if chunk.text:
                    parts.append(chunk.text)
            for structured in ctx.structured:
                if structured:
                    parts.append(structured)

        answer = "\n\n".join(parts) if parts else "(no context retrieved)"
        await memory.add_answer(answer, "passthrough", "/generation/passthrough")
