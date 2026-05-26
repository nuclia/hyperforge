import asyncio
from typing import Optional

import yaml

from hyperforge.engine import main


class ARAG:
    def run(
        self,
        config_file: str = "definition.yaml",
        question: Optional[str] = None,
        intermediate_steps: bool = True,
    ):
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        if question is None:
            question = input("Question to do: ")
        if question is None:
            print("No question detected")
            return
        return asyncio.run(
            main(
                config=config,
                question=question,
            )
        )
