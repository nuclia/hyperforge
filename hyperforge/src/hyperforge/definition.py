from typing import Any, Dict

from pydantic import BaseModel


class NUAConfig(BaseModel):
    key: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    method: str | None = None
    input_schema: Dict[str, Any] | None = None
    lazy_load: bool = False
