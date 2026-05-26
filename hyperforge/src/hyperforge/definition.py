from typing import Any, Dict

from pydantic import BaseModel


class NUAConfig(BaseModel):
    key: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
