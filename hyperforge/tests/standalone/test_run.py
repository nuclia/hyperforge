import json
import subprocess
import sys

import pytest

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


def test_load_hyperforge_builtins_registers_real_agents_and_drivers():
    # Registry configuration is process-global and module decorators only run once.
    # Use a fresh interpreter so this integration check cannot affect other tests.
    script = """
import json

import hyperforge.standalone.run as standalone_run
from hyperforge.configure import GLOBAL_REGISTRY

standalone_run.BUILTIN_MODULES = (
    "hyperforge_static",
    "hyperforge_mcp",
    "hyperforge_mcp.stdio",
)
standalone_run.load_hyperforge_builtins()
print(json.dumps({
    "static": GLOBAL_REGISTRY.context_agents["static"].klass.__module__,
    "mcphttp": GLOBAL_REGISTRY.drivers["mcphttp"].klass.__module__,
    "mcpstdio": GLOBAL_REGISTRY.drivers["mcpstdio"].klass.__module__,
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "static": "hyperforge_static.agent",
        "mcphttp": "hyperforge_mcp.http",
        "mcpstdio": "hyperforge_mcp.stdio",
    }


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
