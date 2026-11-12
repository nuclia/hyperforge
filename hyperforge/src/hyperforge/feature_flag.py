import hashlib
import json
import threading
from functools import cached_property
from typing import Any, Optional

import mrflagly
import pydantic
from pydantic_settings import BaseSettings, SettingsConfigDict

_flag_service = threading.local()


class FlagContext(pydantic.BaseModel):
    """Class to hold information that can then be passed to the feature flag service to determine if a flag is enabled or not.

    Having it as a class allows to simply pass this object down to any function that needs to check for a feature flag and have all the relevant information in one place, instead of having to pass multiple parameters or a dictionary.
    """

    account_id: Optional[str] = None
    kbid: str

    @pydantic.computed_field  # type: ignore[prop-decorator]
    @cached_property
    def account_id_md5(self) -> Optional[str]:
        if self.account_id is not None:
            return hashlib.md5(self.account_id.encode()).hexdigest()
        return None


class Features:
    # RAO
    FILTERED_AGENTS_FEATURE_FLAG = "rao_filtered_agents_{environment}"
    FILTERED_DRIVERS_FEATURE_FLAG = "rao_filtered_drivers_{environment}"
    AUDIT_RAO_ASK_ENDPOINT = "learning_audit_rao_ask"
    RAO_ACCOUNT_ENABLED_FEATURE_FLAG = "rao_account_enabled_{account_md5}"


class FlagService:
    def __init__(self):
        self.settings = Settings()
        if self.settings.flag_settings_url is None:
            self.flag_service = mrflagly.FlagService(data=json.dumps(DEFAULT_FLAG_DATA))  # ty: ignore[unresolved-attribute]
        else:
            self.flag_service = mrflagly.FlagService(  # ty: ignore[unresolved-attribute]
                url=self.settings.flag_settings_url
            )

    def enabled(
        self, flag_key: str, default: bool = False, context: Optional[dict] = None
    ) -> bool:
        if context is None:
            context = {}
        context["environment"] = self.settings.running_environment

        return self.flag_service.enabled(flag_key, default=default, context=context)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_parse_none_str="null")
    running_environment: str = pydantic.Field(
        default="local",
        validation_alias=pydantic.AliasChoices("environment", "running_environment"),
    )
    flag_settings_url: str | None = (
        "https://cdn.rag.progress.cloud/features/features-v2.json"
    )


DEFAULT_FLAG_DATA: dict[str, Any] = {
    # These are just defaults to use for local dev and tests
    Features.FILTERED_AGENTS_FEATURE_FLAG.format(environment="local"): {
        "rollout": 0,
        "variants": {"agents": []},
    },
    Features.FILTERED_DRIVERS_FEATURE_FLAG.format(environment="local"): {
        "rollout": 0,
        "variants": {"drivers": []},
    },
    Features.AUDIT_RAO_ASK_ENDPOINT: {
        "rollout": 0,
        "variants": {"environment": ["local"]},
    },
}


def get_flag_service() -> FlagService:
    if getattr(_flag_service, "service", None) is None:
        _flag_service.service = FlagService()
    return _flag_service.service


def has_feature(
    flag_key: str,
    default: bool = False,
    context: dict[str, str] | None | FlagContext = None,
) -> bool:
    fs = get_flag_service()

    if isinstance(context, FlagContext):
        context_dict = context.model_dump()
    else:
        context_dict = context or {}

    return fs.enabled(flag_key, default=default, context=context_dict)
