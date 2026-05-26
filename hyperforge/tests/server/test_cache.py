from hyperforge.models import HistoryQuestionAnswer
from hyperforge.server.cache import CachedSessionQA, NoCache, ValkeyCache


async def test_noop_cache():
    c = NoCache()

    qa = CachedSessionQA(c, "kbid", "session")
    assert await qa.get() is None
    await qa.append(HistoryQuestionAnswer(question="?", answer="!"))
    assert await qa.get() is None


async def test_valkey_cache(valkey_cache: ValkeyCache):
    my_qa = HistoryQuestionAnswer(question="?", answer="!")

    qa = CachedSessionQA(valkey_cache, "kbid", "session")
    assert await qa.get() is None
    await qa.append(my_qa)
    assert await qa.get() == [my_qa]
