from dataclasses import dataclass
from typing import Any, Sequence, assert_never

from hyperforge.definition import FunctionDefinition
from hyperforge.memory import Context
from nuclia_models.predict.remi import RemiResponse
from pydantic import BaseModel


class WorkerExecutionRequest(BaseModel):
    code: str
    question: str
    local_vars: dict[str, Any]
    global_vars: dict[str, Any]
    function_names: dict[str, dict[str, FunctionDefinition]]


class RestrictedPythonTask(BaseModel):
    function: str
    agent: str
    args: tuple[Any, ...]
    keyword_args: dict[str, Any]


# The types that are visible to the worker in parameters and return types
WorkerTypes = (
    str
    | bool
    | int
    | Context
    | Sequence["WorkerTypes"]
    | dict[str, "WorkerTypes"]
    | None
    | RemiResponse
    | RestrictedPythonTask
)
WorkerModels = {
    "Context": Context,
    "RemiResponse": RemiResponse,
}


def deserialize(msg: Any) -> WorkerTypes:
    if isinstance(msg, dict) and "__model__" in msg:
        model_name = msg["__model__"]
        if model_name == "RestrictedPythonTask":
            # Special case for RestrictedPythonTask because args are type Any and cannot be serialized
            # to the original with the default pydantic serialization/deserialization
            return RestrictedPythonTask(
                function=msg["function"],
                agent=msg["agent"],
                args=tuple(deserialize(a) for a in msg["args"]),
                keyword_args={
                    k: deserialize(v) for k, v in msg["keyword_args"].items()
                },
            )
        else:
            return WorkerModels[model_name].model_validate(msg)  # type: ignore
    elif isinstance(msg, list):
        return [deserialize(m) for m in msg]
    else:
        return msg


def serialize(message: WorkerTypes) -> Any:
    converted: Any
    if isinstance(message, RestrictedPythonTask):
        # Special case for RestrictedPythonTask because args are type Any and cannot be serialized
        # to the original with the default pydantic serialization/deserialization
        converted = {
            "__model__": "RestrictedPythonTask",
            "function": message.function,
            "agent": message.agent,
            "args": [serialize(a) for a in message.args],
            "keyword_args": {k: serialize(v) for k, v in message.keyword_args.items()},
        }
    elif isinstance(message, BaseModel):
        model = message.__class__.__name__
        converted = message.model_dump()
        converted["__model__"] = model
    elif isinstance(message, list):
        converted = [serialize(m) for m in message]
    elif isinstance(message, dict):
        converted = {k: deserialize(v) for k, v in message.items()}
    else:
        converted = message

    return converted


class SandboxMessage:
    # agent -> worker to initiate execution
    @dataclass
    class Run:
        run: WorkerExecutionRequest

    # worker -> agent to run a function
    @dataclass
    class Request:
        task: RestrictedPythonTask

    # agent -> worker with function results
    @dataclass
    class Response:
        result: WorkerTypes

    # worker -> agent when it has finished
    @dataclass
    class Done:
        pass

    # worker -> agent in case of error
    @dataclass
    class Error:
        error: str

    AnyMessage = Run | Request | Response | Done | Error

    @classmethod
    def parse(cls, data: dict[str, Any]) -> AnyMessage:
        match data["_"]:
            case "run":
                return cls.Run(run=WorkerExecutionRequest.model_validate(data["run"]))
            case "request":
                task = deserialize(data["task"])
                assert isinstance(task, RestrictedPythonTask)
                return cls.Request(task=task)
            case "response":
                result = deserialize(data["result"])
                return cls.Response(result=result)
            case "done":
                return cls.Done()
            case "error":
                return cls.Error(error=data["error"])
            case _:
                raise ValueError("Invalid message type")

    @classmethod
    def serialize(cls, message: AnyMessage) -> dict[str, Any]:
        match message:
            case cls.Run():
                return {"_": "run", "run": message.run.model_dump()}
            case cls.Request():
                return {"_": "request", "task": serialize(message.task)}
            case cls.Response():
                return {"_": "response", "result": serialize(message.result)}
            case cls.Done():
                return {"_": "done"}
            case cls.Error():
                return {"_": "error", "error": message.error}
            case _:
                assert_never(message)
