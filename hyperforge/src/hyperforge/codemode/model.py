from dataclasses import dataclass
from typing import Any, Sequence, assert_never, cast

from nuclia_models.predict.remi import RemiResponse
from pydantic import BaseModel

from hyperforge.definition import FunctionDefinition
from hyperforge.memory import Context


class WorkerExecutionRequest(BaseModel):
    code: str
    question: str = ""
    local_vars: dict[str, Any]
    global_vars: dict[str, Any]
    function_names: dict[str, dict[str, FunctionDefinition]]
    max_runtime_seconds: float | None = None
    max_memory_bytes: int | None = None


class RestrictedPythonTask(BaseModel):
    function: str
    agent: str
    args: tuple[Any, ...]
    keyword_args: dict[str, Any]


class WorkerError(BaseModel):
    error: str


WorkerTypes = (
    str
    | bool
    | int
    | float
    | Context
    | Sequence["WorkerTypes"]
    | dict[str, "WorkerTypes"]
    | None
    | RemiResponse
    | RestrictedPythonTask
    | WorkerError
)
WorkerModels: dict[str, type[BaseModel]] = {
    "Context": Context,
    "RemiResponse": RemiResponse,
    "WorkerError": WorkerError,
}


def deserialize(msg: Any) -> WorkerTypes:
    if isinstance(msg, dict) and "__model__" in msg:
        model_name = msg["__model__"]
        if model_name == "RestrictedPythonTask":
            return RestrictedPythonTask(
                function=msg["function"],
                agent=msg["agent"],
                args=tuple(deserialize(a) for a in msg["args"]),
                keyword_args={
                    k: deserialize(v) for k, v in msg["keyword_args"].items()
                },
            )
        model = WorkerModels.get(model_name)
        if model is not None:
            return cast(WorkerTypes, model.model_validate(msg))
    if isinstance(msg, list):
        return [deserialize(m) for m in msg]
    if isinstance(msg, dict):
        return {k: deserialize(v) for k, v in msg.items()}
    return msg


def serialize(message: Any) -> Any:
    if isinstance(message, RestrictedPythonTask):
        return {
            "__model__": "RestrictedPythonTask",
            "function": message.function,
            "agent": message.agent,
            "args": [serialize(a) for a in message.args],
            "keyword_args": {k: serialize(v) for k, v in message.keyword_args.items()},
        }
    if isinstance(message, BaseModel):
        converted = message.model_dump()
        converted["__model__"] = message.__class__.__name__
        return converted
    if isinstance(message, (list, tuple)):
        return [serialize(m) for m in message]
    if isinstance(message, dict):
        return {k: serialize(v) for k, v in message.items()}
    return message


class SandboxMessage:
    @dataclass
    class Run:
        run: WorkerExecutionRequest
        token: str | None = None

    @dataclass
    class Request:
        task: RestrictedPythonTask

    @dataclass
    class Response:
        result: WorkerTypes

    @dataclass
    class Done:
        pass

    @dataclass
    class Error:
        error: str

    AnyMessage = Run | Request | Response | Done | Error

    @classmethod
    def parse(cls, data: dict[str, Any]) -> AnyMessage:
        match data["_"]:
            case "run":
                return cls.Run(
                    run=WorkerExecutionRequest.model_validate(data["run"]),
                    token=data.get("token"),
                )
            case "request":
                task = deserialize(data["task"])
                assert isinstance(task, RestrictedPythonTask)
                return cls.Request(task=task)
            case "response":
                return cls.Response(result=deserialize(data["result"]))
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
                return {
                    "_": "run",
                    "run": message.run.model_dump(),
                    "token": message.token,
                }
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
