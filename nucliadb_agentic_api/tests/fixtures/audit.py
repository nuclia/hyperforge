from collections.abc import AsyncIterator
from unittest.mock import patch

import pytest
from nucliadb_utils.settings import audit_settings
from nucliadb_utils.utilities import Utility
from pytest_mock import MockerFixture

from nucliadb_agentic_api.src.nucliadb_agentic_api.ask.audit import StreamAuditStorage
from nucliadb_agentic_api.tests.fixtures.utils import global_utility


@pytest.fixture(scope="function")
async def audit(
    nats_server: str,
    mocker: MockerFixture,
) -> AsyncIterator[StreamAuditStorage]:
    with (
        patch("nuclia_arag_api.app.start_audit_utility"),
        patch("nuclia_arag_api.app.stop_audit_utility"),
        patch.object(audit_settings, "audit_driver", "stream"),
        patch.object(audit_settings, "audit_jetstream_servers", [nats_server]),
    ):
        audit = StreamAuditStorage(
            [nats_server],
            audit_settings.audit_jetstream_target,  # type: ignore
            audit_settings.audit_partitions,
            audit_settings.audit_hash_seed,
            nats_creds=None,
            service="nuclia_arag_ask.tests",
        )
        await audit.initialize()

        mocker.spy(audit.js, "publish")
        mocker.spy(audit, "send")
        mocker.spy(audit, "retrieve")
        mocker.spy(audit, "ask")

        with global_utility(Utility.AUDIT, audit):
            yield audit

        await audit.finalize()
