from dataclasses import dataclass
from functools import partial
from multiprocessing.connection import Connection
from typing import Any, Dict, List, Optional

from hyperforge.definition import FunctionDefinition
from hyperforge.memory import Chunk
from hyperforge.memory.memory import Context
from RestrictedPython import compile_restricted  # type: ignore
from RestrictedPython.Guards import (  # type: ignore
    guarded_iter_unpack_sequence,
    guarded_unpack_sequence,
    safe_builtins,  # type: ignore
)

from agents.restricted.src.hyperforge_restricted.model import RestrictedPythonTask

EXAMPLES = {
    "discourse": """
import httpx
resp = get("discourse_url", param={q=question}).json()
for message in resp:
    chunks.append(Chunk(chunk_id=message.id, text=message.text, labels=message.labels))
"""
}


# Define allowed builtins and imports
allowed_builtins = {
    **safe_builtins,  # Default safe builtins
    "__import__": lambda name, *args: (
        __import__(name) if name in {"math"} else None
    ),  # Restrict imports
}

# Define restricted globals
restricted_globals = {
    "__builtins__": allowed_builtins,  # Restrict built-ins
}


class PythonAgentWorker:
    context: Optional[Context] = None

    def __init__(self, pipe: Connection, debug: bool = False):
        self.pipe = pipe
        self.functions_agent_id: Dict[str, List[str]] = {}
        self.debug = debug

    def _process_question_context_sync(
        self,
        code: str,
        question: str,
        local_vars: Dict[str, Any],
        global_vars: Dict[str, Any],
        function_names: Dict[str, Dict[str, FunctionDefinition]],
    ):
        try:
            for agent, functions in function_names.items():
                for function in functions.keys():
                    if function in global_vars:
                        self.functions_agent_id[function].append(agent)
                    else:
                        global_vars[function] = partial(
                            self.valid_functions_call, function
                        )
                        self.functions_agent_id[function] = [agent]

            byte_code = compile_restricted(code, filename="<inline code>", mode="exec")

            global_vars.update(
                {
                    "__builtins__": safe_builtins,
                    "_getitem_": lambda obj, index: obj[index],
                    "_getiter_": iter,
                    "dataclass": dataclass,
                    "Chunk": Chunk,
                    "Context": Context,
                    "List": List,
                    "Any": Any,
                    "Dict": Dict,
                    "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
                    "_unpack_sequence_": guarded_unpack_sequence,
                    "save": self.save,
                    "question": question,
                }
            )
            if self.debug:
                global_vars["pdb"] = __import__("pdb")

            exec(
                byte_code,
                global_vars,
                local_vars,
            )
        except Exception as e:
            self.pipe.send(
                RestrictedPythonTask(
                    function="_error",
                    agent="_controller",
                    args=(str(e),),
                    keyword_args={},
                )
            )

        self.pipe.send(
            RestrictedPythonTask(
                function="_close",
                agent="_controller",
                args=(None,),
                keyword_args={},
            )
        )

    def save(
        self,
        contexts: List[Context] = [],
        resolved: bool = True,
        missing: list[str] = [],
        question: str = "",
        final_answer: Optional[str] = None,
    ):
        self.pipe.send(
            RestrictedPythonTask(
                function="save",
                agent="_controller",
                args=(None,),
                keyword_args={
                    "contexts": contexts,
                    "resolved": resolved,
                    "missing": missing,
                    "question": question,
                    "final_answer": final_answer,
                },
            )
        )
        return self.pipe.recv()

    def valid_functions_call(
        self,
        function_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if "agent_id" in kwargs:
            agent_id = kwargs.pop("agent_id")
        else:
            agent_id = None
        if agent_id is not None:
            if agent_id not in self.functions_agent_id.get(function_name, []):
                raise Exception(
                    f"Agent ID ({agent_id}) not authorized for this function {function_name}"
                )
        else:
            agents = self.functions_agent_id.get(function_name, [])
            if len(agents) == 0:
                raise ValueError(f"No agent found for function {function_name}")
            elif len(agents) > 1:
                raise ValueError(
                    f"Multiple agents found for function {function_name}: {agents}, please specify agent_id"
                )

            agent_id = agents[0]

        self.pipe.send(
            RestrictedPythonTask(
                function=function_name, agent=agent_id, args=args, keyword_args=kwargs
            )
        )
        return self.pipe.recv()
