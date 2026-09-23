from collections import OrderedDict

from hyperforge.memory.memory import NoMemorySessionMemory
from hyperforge.models import MemoryConfig
from hyperforge.server.cache import NoCache


def test_question_memory_initializes_independent_generation_rules():
    session = NoMemorySessionMemory(MemoryConfig(), "agent", "default", cache=NoCache())
    session.init("session")

    first_question = session.start_question("First question")
    second_question = session.start_question("Second question")

    assert first_question.generation_rules == OrderedDict()
    assert second_question.generation_rules == OrderedDict()

    first_question.generation_rules["format"] = "Be concise."

    assert second_question.generation_rules == OrderedDict()
