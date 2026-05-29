from nucliadb_models.search import AskRequest, MaxTokens
from pydantic import BaseModel
from typing_extensions import assert_never

from nucliadb_agentic_api.ask.search.parsers.fetcher import (
    Fetcher,
)


class Generation(BaseModel):
    """Request field related with response generation"""

    use_visual_llm: bool
    max_context_tokens: int
    max_answer_tokens: int | None


class _AskParser:
    def __init__(self, kbid: str, item: AskRequest, fetcher: Fetcher):
        self.kbid = kbid
        self.item = item
        self.fetcher = fetcher

    async def parse(self) -> Generation:
        use_visual_llm = await self.fetcher.get_visual_llm_enabled()

        if self.item.max_tokens is None:
            max_tokens = None
        elif isinstance(self.item.max_tokens, int):
            max_tokens = MaxTokens(
                context=None,
                answer=self.item.max_tokens,
            )
        elif isinstance(self.item.max_tokens, MaxTokens):
            max_tokens = self.item.max_tokens
        else:  # pragma: no cover
            assert_never(self.item.max_tokens)

        max_context_tokens = await self.fetcher.get_max_context_tokens(max_tokens)
        max_answer_tokens = self.fetcher.get_max_answer_tokens(max_tokens)

        return Generation(
            use_visual_llm=use_visual_llm,
            max_context_tokens=max_context_tokens,
            max_answer_tokens=max_answer_tokens,
        )


async def parse_ask(
    kbid: str,
    item: AskRequest,
    *,
    fetcher: Fetcher | None = None,
) -> Generation:
    fetcher = fetcher or fetcher_for_ask(kbid, item)
    parser = _AskParser(kbid, item, fetcher)
    return await parser.parse()


def fetcher_for_ask(kbid: str, item: AskRequest) -> Fetcher:
    return Fetcher(
        kbid=kbid,
        query=item.query,
        user_vector=None,
        vectorset=item.vectorset,
        rephrase=item.rephrase,
        rephrase_prompt=None,
        generative_model=item.generative_model,
        query_image=item.query_image,
    )
