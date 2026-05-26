import logging

import fire  # type: ignore
import jinja2

# Jinja vulnerability is not a concern since this is used to format a string prompt, not to render HTML
PROMPT_ENVIRONMENT = jinja2.Environment()  # nosemgrep


logger = logging.getLogger("hyperforge")


def cli():
    from hyperforge.arag import ARAG

    fire.Fire(ARAG)
