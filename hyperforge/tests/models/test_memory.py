from nucliadb_models import CreateResourcePayload, InputConversationField
from nucliadb_models.resource import KnowledgeBoxObj
from nucliadb_models.text import TextField
from nucliadb_sdk import NucliaDB

from hyperforge.api.models import INFO_FIELD_ID
from hyperforge.memory.memory import (
    QUESTION_ANSWERS_FIELD,
    QuestionMemory,
    SessionMemory,
)
from hyperforge.models import (
    Chunk,
    Context,
    MemoryConfig,
    NucliaDBMemoryConfig,
    Rules,
)


async def test_memory_save_load(sdk: NucliaDB, arag_kb: KnowledgeBoxObj):
    config = MemoryConfig(
        nucliadb=NucliaDBMemoryConfig(url=sdk.base_url, kbid=arag_kb.uuid)
    )

    resource = sdk.create_resource(
        kbid=arag_kb.uuid,
        content=CreateResourcePayload(
            title="my session",
            texts={INFO_FIELD_ID: TextField(body="some session info")},
            conversations={QUESTION_ANSWERS_FIELD: InputConversationField()},
        ),
    )
    rid = resource.uuid

    # Initially empty
    memory = SessionMemory.from_config(
        config, "agent", workflow_id="default", rules=Rules(rules=[])
    )
    memory.init(rid)
    result, interactions = await memory.context_history()
    assert result == ""
    assert interactions == 0

    # After a question, it has a Q & A
    q = memory.start_question("How to make bread?")
    await q.add_answer("With ingredients and love", "module", "path")
    await q.add_final_answer()
    await q.save()
    result, interactions = await memory.context_history()
    assert (
        interactions == 1
        and result
        == "- Question: How to make bread?\n- Answer: With ingredients and love\n"
    )
    # After a reload, it still has the Q & A
    memory = SessionMemory.from_config(
        config, "agent", workflow_id="default", rules=Rules(rules=[])
    )
    memory.init(rid)
    result, interactions = await memory.context_history()
    assert (
        interactions == 1
        and result
        == "- Question: How to make bread?\n- Answer: With ingredients and love\n"
    )


def test_contexts_markdown_can_include_summaries_with_chunks():
    session = SessionMemory.from_config(
        MemoryConfig(), "agent", workflow_id="default", rules=Rules(rules=[])
    )
    memory = QuestionMemory(session, "What is the answer?")
    memory.contexts = [
        Context(
            original_question_uuid="question-1",
            actual_question_uuid="question-1",
            question="What is the answer?",
            source="test",
            agent="test",
            summary="The answer is already known from an upstream tool.",
            citations_id="block-AA",
            chunks=[Chunk(chunk_id="chunk-1", text="Supporting source text")],
        )
    ]

    markdown = memory.contexts_markdown(include_summaries=True)

    assert "The answer is already known from an upstream tool." in markdown
    assert "[block-AA-0]" in markdown
    assert "Supporting source text" in markdown
