import re
from dataclasses import dataclass
from typing import Optional

from lingua import LanguageDetectorBuilder

from hyperforge import logger
from hyperforge_nucliadb.ask.config import AskAgentConfig
from hyperforge_nucliadb.ask.kb_analysis import KnowledgeBoxInfo

LID_MODEL = None


def load_lingua() -> None:
    global LID_MODEL
    LID_MODEL = (
        LanguageDetectorBuilder.from_all_languages()
        .with_preloaded_language_models()
        .build()
    )


def detect_language(text: str, default="en") -> str:
    global LID_MODEL
    lang = default

    if len(text.strip()) == 0:
        return default

    if isinstance(text, bytes):
        text = text.decode()

    text = cleanup_text_for_langdetect(text)

    if LID_MODEL is None:
        load_lingua()
    assert LID_MODEL is not None  # for type checker
    result = LID_MODEL.detect_language_of(text.replace("\n", " "))

    if result is not None:
        lang = result.iso_code_639_1.name.lower()

    return lang


def cleanup_text_for_langdetect(text: str) -> str:
    # Get all text from splitting docs
    regex = re.compile(r"(\[image: [a-zA-Z0-9.]*\])", re.S)
    return re.sub(regex, "", text)


@dataclass
class QueryAnalisys:
    translate_need: Optional[str] = None


def pre_query_analysis(
    config: AskAgentConfig, question: str, kb_analysis: KnowledgeBoxInfo
) -> QueryAnalisys:
    # LLM QUERY
    result = QueryAnalisys()

    # Rephrase as semantic to query semantic engine -> custom prompt

    query_language = detect_language(question)

    # Check if the query language is in the languages of the KB
    # or get the most common language in the KB to translate the query
    if query_language in kb_analysis.languages:
        result.translate_need = None
    else:
        try:
            result.translate_need = sorted(
                kb_analysis.languages.items(), key=lambda item: item[1]
            )[0][0]
        except IndexError:
            logger.warning(
                "No languages found in the KB analysis, cannot determine if translation is needed"
            )
            # Kb may not have any languages
            result.translate_need = None

    return result
