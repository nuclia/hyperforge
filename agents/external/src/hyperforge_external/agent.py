from copy import deepcopy
from time import time

from hyperforge.agent import Agent
from hyperforge.configure import agent
from hyperforge.manager import Manager
from hyperforge.memory import QuestionMemory
from hyperforge.trace import trace_agent
from hyperforge.utils import check_dns
from hyperforge.utils.http import safe_http_client

from hyperforge import PROMPT_ENVIRONMENT, logger
from hyperforge_external.config import ExternalCallAgentConfig

EXTERNAL_CALL_PROMPT = """
Collect the parameters to call the query based on the schema and the context:
{{prompt}}


[START OF CONTEXT]
{{context}}
[END OF CONTEXT]

MAIN QUESTION: {{question}}

MAIN ANSWER: {{answer}}
"""

EXTERNAL_CALL_PROMPT_TEMPLATE = PROMPT_ENVIRONMENT.from_string(EXTERNAL_CALL_PROMPT)


@agent(
    id="external",
    agent_type="postprocess",
    title="External Call",
    description="Agent that performs External Call.",
    config_schema=ExternalCallAgentConfig,
)
class ExternalCallAgent(Agent[ExternalCallAgentConfig]):
    config: ExternalCallAgentConfig

    @trace_agent
    async def __call__(
        self,
        memory: QuestionMemory,
        manager: Manager,
    ):
        t0 = time()

        prompt = EXTERNAL_CALL_PROMPT_TEMPLATE.render(
            question=memory.original_question,
            context=memory.contexts_minimal(),
            answer=memory.final_answer,
            prompt=self.config.prompt,
        )
        t0 = time()

        resp = None

        input_nuclia_tokens: float = 0.0
        output_nuclia_tokens: float = 0.0

        evaluation = None
        async with safe_http_client() as client:
            url = await check_dns(self.config.url)

            if self.config.call_schema:
                logger.debug(
                    f"Calling external API with schema: {self.config.call_schema}, description: {self.config.description}"
                )
                evaluation, input_tokens, output_tokens = await manager.execute_json(
                    user_id="external_call_agent",
                    model=self.config.model,
                    schema={
                        "title": "external_call_agent",
                        "description": self.config.description
                        or "Choose the parameters to call an external API",
                        "parameters": self.config.call_schema,
                    },
                    prompt=prompt,
                    tracking=memory.get_tracking_info(),
                )
                output_nuclia_tokens += output_tokens
                input_nuclia_tokens += input_tokens
                logger.debug(f"Json to do the call: {evaluation}")
                resp = await client.request(
                    method=self.config.method.value,
                    url=url,
                    json=evaluation,
                    headers=self.config.headers,
                )
            elif self.config.context:
                logger.debug(
                    f"Calling external API with generated context: {memory.contexts}"
                )
                resp = await client.request(
                    method=self.config.method.value,
                    url=url,
                    json=memory.contexts,
                    headers=self.config.headers,
                )
            else:
                logger.debug(
                    f"Calling external API with call object: {self.config.call_obj}"
                )
                if self.config.call_obj is not None:
                    evaluation = deepcopy(self.config.call_obj)
                else:
                    evaluation = {}
                evaluation["answer"] = memory.final_answer
                evaluation["question"] = memory.original_question
                resp = await client.request(
                    method=self.config.method.value,
                    url=self.config.url,
                    json=evaluation,
                    headers=self.config.headers,
                )
        error_resp = None
        if resp is None:
            error_resp = "No response from external API"
        elif resp.status_code != 200:
            error_resp = f"Error calling external API: {resp.status_code} - {resp.content.decode()}"
        if error_resp:
            logger.error(error_resp)
            raise Exception(error_resp)
        logger.info(f"Response from external API: {resp.content.decode()}")
        await memory.add_step(
            step_module="external",
            step_title=self.step_title(self.config.method.value),
            step_value=self.config.url,
            step_reason=resp.content.decode(),
            timeit=time() - t0,
            step_agent_path=f"/postprocess/{self.config.id if self.config.id else 'default'}",
            input_nuclia_tokens=input_nuclia_tokens,
            output_nuclia_tokens=output_nuclia_tokens,
        )
