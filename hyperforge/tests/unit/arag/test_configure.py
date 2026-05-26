from unittest.mock import Mock, patch

from hyperforge.configure import (
    enabled_agent,
    enabled_driver,
)


def test_enabled_agent():
    agent = Mock(id="test_agent")
    with patch("hyperforge.configure.has_feature") as mock_has_feature:
        for env_check, account_check, enabled in (
            (True, True, True),
            (True, False, False),
            (False, True, True),
            (False, False, True),
        ):
            mock_has_feature.side_effect = [env_check, account_check]
            assert enabled_agent(agent, "env", "account_md5") == enabled


def test_enabled_driver():
    driver = Mock(id="test_driver")
    with patch("hyperforge.configure.has_feature") as mock_has_feature:
        for env_check, account_check, enabled in (
            (True, True, True),
            (True, False, False),
            (False, True, True),
            (False, False, True),
        ):
            mock_has_feature.side_effect = [env_check, account_check]
            assert enabled_driver(driver, "env", "account_md5") == enabled
