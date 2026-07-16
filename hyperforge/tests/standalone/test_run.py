import pytest

from hyperforge.standalone.run import load_default_modules


def test_load_default_modules_uses_nuclia_when_available(monkeypatch):
    scanned: list[str] = []
    loaded: list[str] = []

    def fake_scan(path: str):
        scanned.append(path)

    def fake_load_all_configurations(module_name: str):
        loaded.append(module_name)

    # Patch symbols imported inside load_default_modules.
    import hyperforge.configure as configure

    monkeypatch.setattr(configure, "scan", fake_scan)
    monkeypatch.setattr(
        configure, "load_all_configurations", fake_load_all_configurations
    )

    result = load_default_modules()

    assert result == "nuclia_agents"
    assert scanned[:2] == ["nuclia_agents.agents.agents", "nuclia_agents.drivers.drivers"]
    assert loaded == ["nuclia_agents"]


def test_load_default_modules_falls_back_when_nuclia_missing(monkeypatch):
    scanned: list[str] = []
    loaded: list[str] = []

    def fake_scan(path: str):
        scanned.append(path)
        if path.startswith("nuclia_agents"):
            raise ModuleNotFoundError("No module named 'nuclia_agents'")

    def fake_load_all_configurations(module_name: str):
        loaded.append(module_name)

    import hyperforge.configure as configure

    monkeypatch.setattr(configure, "scan", fake_scan)
    monkeypatch.setattr(
        configure, "load_all_configurations", fake_load_all_configurations
    )

    result = load_default_modules()

    assert result == "hyperforge"
    assert "hyperforge" in scanned
    assert loaded == ["hyperforge"]


def test_load_default_modules_does_not_swallow_unrelated_missing_modules(monkeypatch):
    def fake_scan(path: str):
        if path == "nuclia_agents.agents.agents":
            err = ModuleNotFoundError("No module named 'yaml'")
            err.name = "yaml"
            raise err

    def fake_load_all_configurations(module_name: str):
        return None

    import hyperforge.configure as configure

    monkeypatch.setattr(configure, "scan", fake_scan)
    monkeypatch.setattr(
        configure, "load_all_configurations", fake_load_all_configurations
    )

    with pytest.raises(ModuleNotFoundError) as exc:
        load_default_modules()
    assert exc.value.name == "yaml"
