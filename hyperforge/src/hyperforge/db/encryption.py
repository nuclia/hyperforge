import base64
import os
from functools import cache
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from hyperforge.db.settings import EncryptionSettings
from hyperforge.driver import DriverConfig, EncryptedPayload


@cache
def get_fernet() -> Fernet:
    settings = EncryptionSettings()
    return Fernet(settings.encryption_secret_key)


def encrypt_data(data: str) -> str:
    f = get_fernet()
    return f.encrypt(data.encode()).decode()


def decrypt_data(data: str) -> str:
    f = get_fernet()
    try:
        return f.decrypt(token=data, ttl=None).decode()
    except InvalidToken:
        raise ValueError("Invalid encryption token.")


def fernet_key_from_passphrase(
    passphrase: str, salt: bytes | None
) -> tuple[bytes, bytes]:
    """Generate a Fernet key from a passphrase and salt (if provided).
    From https://cryptography.io/en/latest/fernet/#using-passwords-with-fernet
    """
    if salt is None:
        salt = os.urandom(16)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=1_200_000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))
    return key, salt


def dump_without_encrypted_fields(
    model: DriverConfig[EncryptedPayload],
) -> dict[str, Any]:
    data = model.model_dump()  # type: ignore
    for field in model.config.encrypted_fields:  # type: ignore
        if field in data["config"]:
            del data["config"][field]
    return {"config": data}


def encrypt_fields(model: EncryptedPayload) -> dict[str, Any]:
    data = model.model_dump()  # type: ignore
    for field in model.encrypted_fields:  # type: ignore
        if field in data:
            if isinstance(data[field], str):
                data[field] = encrypt_data(data=data[field])
            elif isinstance(data[field], dict):
                for k, v in data[field].items():
                    if isinstance(v, str):
                        data[field][k] = encrypt_data(data=v)
    return data


def decrypt_fields(model: EncryptedPayload) -> None:
    for field in model.encrypted_fields:
        if not hasattr(model, field):
            raise AttributeError(f"Field '{field}' not found in {type(model).__name__}")
        value = getattr(model, field)
        if value is not None:
            try:
                if isinstance(value, str):
                    value = decrypt_data(data=value)
                elif isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, str):
                            value[k] = decrypt_data(data=v)
            except ValueError:
                # We ignore the error to support current unencrypted data
                continue

            setattr(model, field, value)
