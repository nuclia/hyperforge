from typing import Any, ClassVar, Generic, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field


class EncryptedPayload(BaseModel):
    encrypted_fields: ClassVar[list[str]] = []


T = TypeVar("T", bound="EncryptedPayload")


class DriverConfig(BaseModel, Generic[T]):
    id: str | None = None
    identifier: str
    name: str
    provider: Any = Field(
        ..., description="The type of driver, e.g., 'google', 'marklogic', etc."
    )
    config: T = Field(..., description="The configuration specific to the driver.")


class Driver(BaseModel):
    name: str
    provider: str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    async def init(cls, driver: Any) -> Self:
        raise NotImplementedError()
