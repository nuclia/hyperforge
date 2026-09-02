import os
import sys
from dataclasses import dataclass
from functools import partial
from multiprocessing.connection import Connection
from operator import (
    add,
    and_,
    floordiv,
    lshift,
    matmul,
    mod,
    mul,
    or_,
    pow,
    rshift,
    sub,
    truediv,
    xor,
)
from typing import Any, Dict, List, Optional

from RestrictedPython import compile_restricted
from RestrictedPython.Guards import (
    guarded_iter_unpack_sequence,
    guarded_unpack_sequence,
    safe_builtins,
)

from hyperforge.definition import FunctionDefinition
from hyperforge.memory import Chunk
from hyperforge.memory.memory import Context

from .model import RestrictedPythonTask, WorkerError

BLOCKED_EXCEPTIONS = {
    "BaseException",
    "GeneratorExit",
    "KeyboardInterrupt",
    "SystemExit",
}


def _set_memory_limit(max_memory_bytes: int | None) -> None:
    if max_memory_bytes is None or sys.platform == "darwin":
        return
    try:
        import resource

        with open("/proc/self/statm", encoding="ascii") as statm:
            current_virtual_bytes = int(statm.read().split()[0]) * os.sysconf(
                "SC_PAGE_SIZE"
            )
        address_space_limit = current_virtual_bytes + max_memory_bytes
        resource.setrlimit(
            resource.RLIMIT_AS,
            (address_space_limit, address_space_limit),
        )
    except (ImportError, OSError, ValueError, IndexError) as exc:
        raise RuntimeError("Codemode memory limits are unavailable") from exc


def _harden_process(max_memory_bytes: int | None) -> None:
    os.environ.clear()
    os.umask(0o077)
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, OSError, ValueError):
        pass

    if sys.platform.startswith("linux"):
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
            raise RuntimeError("Unable to disable worker privilege escalation")
        libc.prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE

    _set_memory_limit(max_memory_bytes)


def guarded_inplace(operator: str, left: Any, right: Any) -> Any:
    operations = {
        "+=": add,
        "-=": sub,
        "*=": mul,
        "/=": truediv,
        "//=": floordiv,
        "%=": mod,
        "**=": pow,
        "<<=": lshift,
        ">>=": rshift,
        "&=": and_,
        "^=": xor,
        "|=": or_,
        "@=": matmul,
    }
    operation = operations.get(operator)
    if operation is None:
        raise ValueError(f"Unsupported inplace operator: {operator}")
    return operation(left, right)


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
        max_memory_bytes: int | None = None,
    ):
        try:
            _harden_process(max_memory_bytes)
            for agent, functions in function_names.items():
                for function in functions:
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
                    "__builtins__": {
                        **{
                            name: value
                            for name, value in safe_builtins.items()
                            if name not in BLOCKED_EXCEPTIONS
                        },
                        "sum": sum,
                    },
                    "_getitem_": lambda obj, index: obj[index],
                    "_getiter_": iter,
                    "_inplacevar_": guarded_inplace,
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
            exec(byte_code, global_vars, local_vars)
        except BaseException as exc:
            self.pipe.send(
                RestrictedPythonTask(
                    function="_error",
                    agent="_controller",
                    args=(str(exc),),
                    keyword_args={},
                )
            )
        finally:
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
        contexts: List[Context] | None = None,
        resolved: bool = True,
        missing: list[str] | None = None,
        question: str = "",
        final_answer: Optional[str] = None,
    ):
        self.pipe.send(
            RestrictedPythonTask(
                function="save",
                agent="_controller",
                args=(None,),
                keyword_args={
                    "contexts": contexts or [],
                    "resolved": resolved,
                    "missing": missing or [],
                    "question": question,
                    "final_answer": final_answer,
                },
            )
        )
        return self._receive()

    def valid_functions_call(
        self, function_name: str, *args: Any, **kwargs: Any
    ) -> Any:
        agent_id = kwargs.pop("agent_id", None)
        if agent_id is not None:
            if agent_id not in self.functions_agent_id.get(function_name, []):
                raise ValueError(
                    f"Agent ID ({agent_id}) not authorized for this function {function_name}"
                )
        else:
            agents = self.functions_agent_id.get(function_name, [])
            if not agents:
                raise ValueError(f"No agent found for function {function_name}")
            if len(agents) > 1:
                raise ValueError(
                    f"Multiple agents found for function {function_name}: {agents}, "
                    "please specify agent_id"
                )
            agent_id = agents[0]

        self.pipe.send(
            RestrictedPythonTask(
                function=function_name,
                agent=agent_id,
                args=args,
                keyword_args=kwargs,
            )
        )
        return self._receive()

    def _receive(self) -> Any:
        result = self.pipe.recv()
        if isinstance(result, WorkerError):
            raise RuntimeError(result.error)
        return result
