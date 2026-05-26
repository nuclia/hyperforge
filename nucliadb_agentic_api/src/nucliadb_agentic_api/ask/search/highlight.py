import logging
import re
import string

logger = logging.getLogger(__name__)
PRE_WORD = string.punctuation + " "


def highlight_paragraph(
    text: str, words: list[str] | None = None, ematches: list[str] | None = None
) -> str:
    """
    Highlight `text` with <mark></mark> tags around the words in `words` and `ematches`.

    Parameters:
    - text: The text to highlight.
    - words: A list of words to highlight.
    - ematches: A list of exact matches to highlight.

    Returns:
    - The highlighted text.
    """
    REGEX_TEMPLATE = r"(^|\s)({text})(\s|$)"
    text_lower = text.lower()

    marks = [0] * (len(text_lower) + 1)
    ematches = ematches or []
    for quote in ematches:
        quote_regex = REGEX_TEMPLATE.format(text=re.escape(quote.lower()))
        try:
            for match in re.finditer(quote_regex, text_lower):
                start, end = match.span(2)
                marks[start] = 1
                marks[end] = 2
        except re.error:
            logger.warning(
                f"Regex errors while highlighting text. Regex: {quote_regex}"
            )
            continue

    words = words or []
    for word in words:
        word_regex = REGEX_TEMPLATE.format(text=re.escape(word.lower()))
        try:
            for match in re.finditer(word_regex, text_lower):
                start, end = match.span(2)
                if marks[start] == 0 and marks[end] == 0:
                    marks[start] = 1
                    marks[end] = 2
        except re.error:
            logger.warning(f"Regex errors while highlighting text. Regex: {word_regex}")
            continue

    new_text = ""
    actual = 0
    mod = 0
    skip = False

    length = len(text)

    for index, pos in enumerate(marks):
        if skip:
            skip = False
            continue
        if (index - mod) >= length:
            char_pos = ""
        else:
            begining = True
            if index > 0 and text[index - mod - 1] not in PRE_WORD:
                begining = False
            char_pos = text[index - mod]
            if text[index - mod].lower() != text_lower[index]:
                # May be incorrect positioning due to unicode lower
                mod += 1
                skip = True
        if pos == 1 and actual == 0 and begining:
            new_text += "<mark>"
            new_text += char_pos
            actual = 1
        elif pos == 2 and actual == 1:
            new_text += "</mark>"
            new_text += char_pos
            actual = 0
        elif pos == 1 and actual > 0:
            new_text += char_pos
            actual += 1
        elif pos == 2 and actual > 1:
            new_text += char_pos
            actual -= 1
        else:
            new_text += char_pos

    return new_text
