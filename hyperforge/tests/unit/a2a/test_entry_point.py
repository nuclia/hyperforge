from importlib.metadata import distribution


def test_a2a_grpc_console_script_resolves():
    scripts = {
        entry_point.name: entry_point
        for entry_point in distribution("hyperforge").entry_points
        if entry_point.group == "console_scripts"
    }

    entry_point = scripts["hyperforge-a2a-grpc"]

    assert entry_point.value == "hyperforge.a2a.run:run"
    assert entry_point.load() is not None
