from time import time
from typing import Any, ClassVar, Dict, List, Optional
from uuid import uuid4

from hyperforge.agent import Agent
from hyperforge.configure import agent
from hyperforge.context.agent import ContextAgent
from hyperforge.definition import FunctionDefinition
from hyperforge.manager import Manager
from hyperforge.memory import Chunk, Context, QuestionMemory

from hyperforge_a2a.client import (
    build_a2a_client,
    build_send_request,
    collect_text_from_stream_response,
)
from hyperforge_a2a.config import A2AAgentConfig


@agent(
    id="a2a",
    agent_type="context",
    title="A2A Client",
    description="Queries an external agent over the Agent2Agent (A2A) gRPC protocol.",
    config_schema=A2AAgentConfig,
)
class A2AClientAgent(ContextAgent, Agent[A2AAgentConfig]):
    config: A2AAgentConfig
    __published_functions__: ClassVar[Dict[str, FunctionDefinition]] = {
        "a2a_query": FunctionDefinition(
            name="a2a_query",
            description=(
                "Queries an external A2A agent to gather context to answer questions."
            ),
            parameters={
                "question": {
                    "type": "string",
                    "description": "The question to send to the external A2A agent.",
                },
            },
        )
    }

    def _build_metadata(self, memory: QuestionMemory) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if self.config.remote_account:
            metadata["account"] = self.config.remote_account
        if self.config.remote_agent_id:
            metadata["agent_id"] = self.config.remote_agent_id
        metadata["workflow_id"] = self.config.remote_workflow_id

        headers: dict[str, str] = {}
        for header in self.config.valid_headers:
            if header in memory.headers:
                headers[header] = memory.headers[header]
        if headers:
            metadata["headers"] = headers

        metadata.update(self.config.extra_metadata)
        return metadata

    async def a2a_query(
        self,
        question: str,
        memory: QuestionMemory,
        manager: Manager,
        question_uuid: Optional[str] = None,
    ) -> Context:
        t0 = time()
        texts: List[str] = []

        client = await build_a2a_client(self.config.source, self.config.use_tls)
        try:
            request = build_send_request(question, self._build_metadata(memory))
            async for response in client.send_message(request):
                texts.extend(collect_text_from_stream_response(response))
        finally:
            await client.close()

        answer = "\n".join(t for t in texts if t)

        context = Context(
            agent_id=self.agent_id,
            original_question_uuid=memory.original_question_uuid,
            actual_question_uuid=question_uuid,
            question=question,
            title=self.config.title or "A2A remote agent",
            source="a2a",
            agent=self.config.module,
        )
        if answer:
            context.chunks.append(
                Chunk(
                    chunk_id=uuid4().hex,
                    text=answer,
                    labels=[],
                    origin_agent=self.config.module,
                )
            )
        context.summary = answer

        await memory.add_step(
            step_module=self.config.module,
            step_title=self.step_title("A2A response"),
            step_reason="",
            step_agent_path=f"/context/{self.config.id or self.agent_id}",
            step_value=answer,
            timeit=time() - t0,
            input_nuclia_tokens=0,
            output_nuclia_tokens=0,
        )
        return context

    async def _get_question_context(
        self,
        memory: QuestionMemory,
        manager: Manager,
        question_uuid: str,
        question: str,
        flow_id: str,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[str, str]]:
        context = await self.a2a_query(
            question,
            memory,
            manager,
            question_uuid=question_uuid,
        )
        missing = await self.save_ctx_and_return_missing(
            context=context,
            question=question,
            memory=memory,
            manager=manager,
            flow_id=flow_id,
        )
        return [missing] if missing is not None else []
