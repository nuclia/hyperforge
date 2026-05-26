import logging
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

from nucliadb_utils.utilities import MAIN

logger = logging.getLogger("fixtures.utils")


@contextmanager
def global_utility(name: str, util: Any):
    """Hacky set_utility used in tests to provide proper setup/cleanup of utilities.

    Tests can sometimes mess with global state. While fixtures add/remove global
    utilities, component lifecycles do the same. Sometimes, we can left
    utilities unclean or overwrite utilities. This context manager allows tests
    to remove utilities letting the previously set one.

    """

    if name in MAIN:
        logger.debug(
            f"Overwriting previously set utility {name}: {MAIN[name]} with {util}"
        )

    with patch.dict(MAIN, values={name: util}, clear=False):
        yield util
