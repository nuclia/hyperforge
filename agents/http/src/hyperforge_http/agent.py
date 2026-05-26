from time import time
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from hyperforge.agent import Agent
from hyperforge.configure import agent
from hyperforge.context.agent import ContextAgent
from hyperforge.manager import Manager
from hyperforge.memory import Chunk, Context, QuestionMemory
from hyperforge.utils import check_dns
from hyperforge.utils.http import safe_http_client

from hyperforge_http.config import HTTPStaticAgentConfig


@agent(
    id="http",
    agent_type="context",
    title="HTTP Static",
    description="Use HTTP Static to get information from the internet to answer questions.",
    config_schema=HTTPStaticAgentConfig,
)
class HTTPStaticAgent(ContextAgent, Agent[HTTPStaticAgentConfig]):
    async def _get_question_context(
        self,
        memory: QuestionMemory,
        manager: Manager,
        question_uuid: str,
        question: str,
        flow_id: str,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, str]]:
        t0 = time()
        error = None

        url = await check_dns(self.config.url)

        async with safe_http_client() as client:
            if self.config.method == "GET":
                response = await client.get(
                    url,
                    headers=self.config.headers,
                    timeout=self.config.timeout,
                    params={self.config.question_query_param: question}
                    if self.config.question_query_param
                    else None,
                )

            elif self.config.method == "POST":
                response = await client.post(
                    url,
                    headers=self.config.headers,
                    timeout=self.config.timeout,
                    params={self.config.question_query_param: question}
                    if self.config.question_query_param
                    else None,
                    data={self.config.question_post_field: question}
                    if self.config.question_post_field
                    else None,
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {self.config.method}")

        if response.status_code != 200:
            error = f"HTTP request failed with status code {response.status_code}"
            context_text = ""
        else:
            context_text = response.content.decode("utf-8")

        context = Context(
            agent_id=self.config.id if self.config.id else "http",
            original_question_uuid=memory.original_question_uuid,
            actual_question_uuid=question_uuid,
            question=question,
            source="http",
            agent="http",
            title=self.config.title if self.config.title else "HTTP Call",
        )
        context.chunks.append(
            Chunk(
                chunk_id=uuid4().hex,
                text=context_text,
                origin_agent=self.config.module,
            )
        )

        await memory.add_step(
            step_module=self.config.module,
            step_title=self.step_title("HTTP request"),
            step_agent_path=f"/context/{self.config.id if self.config.id else 'default'}",
            step_value=f" HTTP {self.config.method} to {url}",
            timeit=time() - t0,
            input_nuclia_tokens=0,
            output_nuclia_tokens=0,
            error=error,
        )
        missing = await self.save_ctx_and_return_missing(
            context=context,
            question=question,
            memory=memory,
            manager=manager,
            flow_id=flow_id,
        )
        return [missing] if missing is not None else []
