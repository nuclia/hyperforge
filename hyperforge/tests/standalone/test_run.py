import sys

import pytest

from hyperforge.configure import GLOBAL_REGISTRY, clear
from hyperforge.standalone.run import BUILTIN_MODULES, load_default_modules


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
    assert scanned[:2] == [
        "nuclia_agents.agents.agents",
        "nuclia_agents.drivers.drivers",
    ]
    assert loaded == ["nuclia_agents"]


def test_load_default_modules_falls_back_when_nuclia_missing(monkeypatch):
    # Force the decorator in this real built-in module to run even if another test
    # imported it previously, and limit the test to packages always in the workspace.
    monkeypatch.setattr(
        "hyperforge.standalone.run.BUILTIN_MODULES",
        ("hyperforge_static", "hyperforge_mcp", "hyperforge_mcp.stdio"),
    )
    for module in tuple(sys.modules):
        if module.split(".")[0] in {"hyperforge_static", "hyperforge_mcp"}:
            monkeypatch.delitem(sys.modules, module)

    clear()
    GLOBAL_REGISTRY.clear()
    try:
        result = load_default_modules()

        assert result == "hyperforge"
        assert "static" in GLOBAL_REGISTRY.context_agents
        registration = GLOBAL_REGISTRY.context_agents["static"]
        assert registration.klass.__module__ == "hyperforge_static.agent"
        assert GLOBAL_REGISTRY.drivers["mcphttp"].klass.__module__ == (
            "hyperforge_mcp.http"
        )
        assert GLOBAL_REGISTRY.drivers["mcpstdio"].klass.__module__ == (
            "hyperforge_mcp.stdio"
        )
    finally:
        clear()
        GLOBAL_REGISTRY.clear()


def test_builtin_modules_include_agents_and_drivers():
    assert "hyperforge_static" in BUILTIN_MODULES
    assert "hyperforge_mcp" in BUILTIN_MODULES


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
