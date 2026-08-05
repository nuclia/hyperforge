import pytest
from hyperforge.db.encryption import dump_without_encrypted_fields, encrypt_fields

from hyperforge_a2a.config_driver import A2ADriverConfig, A2AInnerConfig
from hyperforge_a2a.driver import A2ADriver


def test_a2a_driver_config_declares_sensitive_fields():
    assert A2AInnerConfig.encrypted_fields == [
        "authorization",
        "client_private_key",
    ]


def test_a2a_driver_config_validates_tls_pair():
    with pytest.raises(ValueError, match="configured together"):
        A2AInnerConfig(
            endpoint="a2a.example.com:443",
            use_tls=True,
            client_certificate_chain="certificate",
        )

    with pytest.raises(ValueError, match="requires an HTTPS"):
        A2AInnerConfig(endpoint="http://a2a.example.com", use_tls=True)


async def test_a2a_driver_initializes_from_config():
    config = A2ADriverConfig(
        identifier="remote-a2a",
        name="Remote A2A",
        provider="a2a",
        config=A2AInnerConfig(
            endpoint="a2a.example.com:443",
            use_tls=True,
            authorization="Bearer secret",
        ),
    )

    driver = await A2ADriver.init(config)

    assert driver.name == "Remote A2A"
    assert driver.config.endpoint == "a2a.example.com:443"
    assert driver.config.authorization == "Bearer secret"


def test_a2a_driver_encrypts_and_omits_secrets(monkeypatch):
    monkeypatch.setattr(
        "hyperforge.db.encryption.encrypt_data", lambda data: f"encrypted:{data}"
    )
    inner_config = A2AInnerConfig(
        endpoint="a2a.example.com:443",
        use_tls=True,
        authorization="Bearer secret",
        client_certificate_chain="certificate",
        client_private_key="private-key",
    )
    driver_config = A2ADriverConfig(
        identifier="remote-a2a",
        name="Remote A2A",
        provider="a2a",
        config=inner_config,
    )

    encrypted = encrypt_fields(inner_config)
    public = dump_without_encrypted_fields(driver_config)["config"]["config"]

    assert encrypted["authorization"] == "encrypted:Bearer secret"
    assert encrypted["client_private_key"] == "encrypted:private-key"
    assert encrypted["client_certificate_chain"] == "certificate"
    assert "authorization" not in public
    assert "client_private_key" not in public
    assert public["client_certificate_chain"] == "certificate"


@pytest.mark.parametrize(
    ("delegated", "expected"),
    [
        ("Bearer delegated", "Bearer delegated"),
        (None, "Bearer static"),
    ],
)
async def test_a2a_driver_authorization_precedence(monkeypatch, delegated, expected):
    captured = {}

    async def fake_build_client(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("hyperforge_a2a.driver.build_a2a_client", fake_build_client)
    driver = A2ADriver(
        name="Remote A2A",
        provider="a2a",
        config=A2AInnerConfig(
            endpoint="a2a.example.com:443",
            authorization="Bearer static",
        ),
    )

    await driver.client(authorization=delegated)

    assert captured["authorization"] == expected
