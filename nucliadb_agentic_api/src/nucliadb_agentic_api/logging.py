from importlib.metadata import version
from typing import Optional

import sentry_sdk
from sentry_sdk.integrations.excepthook import ExcepthookIntegration


def set_sentry(zone: str, environment: str, sentry_url: Optional[str] = None):
    if sentry_url:
        sentry_exception = ExcepthookIntegration(always_run=True)
        version_num = version("nucliadb_agentic_api")
        sentry_sdk.init(
            release=version_num,
            environment=environment,
            dsn=sentry_url,
            integrations=[sentry_exception],
        )
        sentry_sdk.set_tag("zone", zone)
