import asyncio
from time import time
from typing import List, cast

from hyperforge import PROMPT_ENVIRONMENT
from hyperforge.manager import Manager
from hyperforge.memory.memory import QuestionMemory
from hyperforge.models import Source

from hyperforge_nucliadb.driver import NucliaDBDriver
from hyperforge_nucliadb.sync.driver import SyncDriver

MULTI_KB_PROMPT = """

You have multiple Knowledge Boxes (KBs) that you can use as data sources to ask your questions.
Every KB has an ID, a description, a set of labels, and  facets that characterize its contents.
Your task is to figure out which KB is the most relevant to a given question by applying the important rules specified.

 Here are the details of each KB:

{% for source in sources.values() %}
{%- set sid = source.id -%}
{%- set sdescription = source.description -%}
{%- set slabels = source.labels -%}
# ID: {{sid}}

{{sdescription}}

labels: {{slabels}}

{% endfor -%}

And given the question: {{question}}

# Important rules to follow

{% for rule in rules %}
{{rule}}
{% endfor -%}

Now, provide your answer following the answer rules:
- Which KB is the most relevant and why(short explanation)?
"""

MULTI_KB_PROMPT_TEMPLATE = PROMPT_ENVIRONMENT.from_string(MULTI_KB_PROMPT)

MULTI_KB_SCHEMA = {
    "title": "source_choice",
    "description": "Choose which KBs should be used to answer the question of the user and why",
    "parameters": {
        "type": "object",
        "properties": {
            "sources": {
                "type": "array",
                "description": "the ids of the choosen sources",
                "items": {"type": "string"},
            },
            "reason": {
                "type": "string",
                "description": "reasoning behind the decision",
            },
        },
    },
}


async def choose_source(
    memory: QuestionMemory,
    manager: Manager,
    sources: List[str],
    question: str,
    ident: str,
    choose_source_model: str = "chatgpt-o3-mini",
    step_title: str = "Knowledge Box Ask: Choose sources",
) -> list[Source]:
    if len(sources) == 0:
        raise Exception("No sources available")

    sources_objects = {}
    for source_id in sources:
        source = await memory.get_session_source(source_id)
        if source is None:
            source = await load_source_information(source_id, manager)
            await memory.set_session_source(source)
        sources_objects[source_id] = source

    # Only one source, try to use it without involving the LLM
    if len(sources) == 1:
        return [sources_objects[sources[0]]]

    t0 = time()
    prompt = MULTI_KB_PROMPT_TEMPLATE.render(
        sources=sources_objects,
        rules=memory.get_rules(),
        question=question,
    )
    data, input, output = await manager.execute_json(
        prompt=prompt,
        schema=MULTI_KB_SCHEMA,
        user_id="source_models",
        model=choose_source_model,
        tracking=memory.get_tracking_info(),
    )

    chosen_sources = cast(list[str], data.get("sources"))
    reason = data.get("reason")

    await memory.add_step(
        step_module="router",
        step_title=step_title,
        step_value=str(chosen_sources),
        step_reason=reason,
        timeit=time() - t0,
        input_nuclia_tokens=input,
        output_nuclia_tokens=output,
        step_agent_path=f"/context/{ident if ident else 'default'}",
    )

    return [sources_objects[chosen] for chosen in chosen_sources]


async def load_source_information(source: str, manager: Manager) -> Source:
    driver = manager.drivers.get(source)
    if not isinstance(driver, NucliaDBDriver) and not isinstance(driver, SyncDriver):
        raise ValueError("Source is not a KnowledgeBox source")
    (
        description,
        labels,
        facets_native,
        paragraph_facets,
        learning_configuration,
    ) = await asyncio.gather(
        driver.description(),
        driver.labels(),
        driver.facets_native(),
        driver.paragraph_facets(),
        driver.get_learning_configuration(),
    )
    return Source(
        id=source,
        description=description,
        labels=labels,
        facets_native=facets_native,
        paragraph_facets=paragraph_facets,
        learning_configuration=learning_configuration,
    )
