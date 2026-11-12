import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import ValidationError

from hyperforge.procedural.cli import main
from hyperforge.procedural.evolution import EvaluationTask, ScoredRollout, evolve
from hyperforge.procedural.graph import GraphEdits, ProceduralGraph, prepare_candidate


def read_json(path):
    return json.loads(path.read_text())


@pytest.fixture
def graph():
    return ProceduralGraph.skeleton()


@pytest.fixture
def kwargs(tmp_path, graph):
    async def runner(graph, task):
        return ScoredRollout(
            task_id=task.id,
            score=0.5,
            trajectory=[{"query": task.query}],
            output=f"output:{task.query}",
        )

    async def refiner(graph, context, rejections):
        return GraphEdits()

    return {
        "initial_graph": graph,
        "output": tmp_path / "run",
        "train_tasks": [EvaluationTask(id="train", query="training prompt")],
        "validation_tasks": [
            EvaluationTask(id="validation", query="SECRET validation prompt")
        ],
        "test_tasks": [EvaluationTask(id="test", query="SECRET test prompt")],
        "runner": runner,
        "refiner": refiner,
        "available_tools": [],
        "provenance": {
            "model": "fake-refiner-v1",
            "evaluator": "fake-score-v1",
            "tools": {},
        },
        "rounds": 1,
    }


@pytest.mark.parametrize(
    "score", [-0.1, 1.1, float("nan"), float("inf"), float("-inf")]
)
def test_score_validation(score):
    with pytest.raises(ValidationError):
        ScoredRollout(task_id="task", score=score)


def test_models_forbid_extra_and_have_independent_defaults():
    with pytest.raises(ValidationError):
        EvaluationTask(id="task", query="q", unknown=True)
    with pytest.raises(ValidationError):
        ScoredRollout(task_id="task", score=0.5, unknown=True)
    first = ScoredRollout(task_id="one", score=0)
    first.trajectory.append({"event": "one"})
    assert ScoredRollout(task_id="two", score=1).trajectory == []
    assert EvaluationTask(id="task", query="q").expected is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "candidate_score, status", [(0.8, "accepted"), (0.5, "accepted"), (0.2, "rejected")]
)
async def test_acceptance_and_cached_baseline(kwargs, candidate_score, status):
    validation_calls = 0
    parents = []

    async def runner(graph, task):
        nonlocal validation_calls
        if task.id == "validation":
            validation_calls += 1
            score = 0.5 if validation_calls == 1 else candidate_score
        else:
            score = 0.1
            parents.append(graph.fingerprint)
        return ScoredRollout(task_id=task.id, score=score)

    kwargs.update(runner=runner, rounds=2)
    result = await evolve(**kwargs)
    output = kwargs["output"]
    assert validation_calls == 3  # Initial baseline once, then each proposal once.
    assert parents == [kwargs["initial_graph"].fingerprint] * 2
    assert result == kwargs["initial_graph"]
    assert read_json(output / "rounds/000001.json")["status"] == status
    pointer = read_json(output / "retained.json")
    assert pointer["validation_score"] == (
        candidate_score if status == "accepted" else 0.5
    )
    assert pointer["round"] == (2 if status == "accepted" else 0)
    assert len(list((output / "rejections").glob("*.json"))) == (
        2 if status == "rejected" else 0
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "proposal", [{"unknown": []}, {"delete_nodes": ["missing"]}, "not edits", object()]
)
async def test_invalid_proposals_do_not_validate(kwargs, proposal):
    calls = []
    original_runner = kwargs["runner"]

    async def runner(graph, task):
        calls.append(task.id)
        return await original_runner(graph, task)

    async def refiner(graph, context, rejections):
        return proposal

    kwargs.update(runner=runner, refiner=refiner)
    assert await evolve(**kwargs) == kwargs["initial_graph"]
    assert calls == ["validation", "train"]
    record = read_json(kwargs["output"] / "rejections/000001.json")
    assert record["reason"] == "proposal_error"
    assert record["validation"]["status"] == "not_run"
    assert record["edits"] is not None
    assert read_json(kwargs["output"] / "retained.json")["validation_score"] == 0.5


@pytest.mark.asyncio
async def test_manifest_and_immutable_artifacts_and_collision(kwargs):
    await evolve(**kwargs)
    output = kwargs["output"]
    manifest = read_json(output / "manifest.json")
    assert manifest["splits"] == {
        "train": ["train"],
        "validation": ["validation"],
        "test": ["test"],
    }
    assert manifest["provenance"] == kwargs["provenance"]
    assert manifest["initial_fingerprint"] == kwargs["initial_graph"].fingerprint
    record = read_json(output / "rounds/000001.json")
    assert record["candidate"] == kwargs["initial_graph"].model_dump(mode="json")
    assert record["edits"] == GraphEdits().model_dump(mode="json")
    assert record["train_traces"][0]["score"] == 0.5
    assert record["validation"]["traces"][0]["task_id"] == "validation"
    before = {p: p.read_bytes() for p in output.rglob("*.json")}
    for path in before:
        if path.name != "retained.json":
            assert path.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        await evolve(**kwargs)
    assert before == {p: p.read_bytes() for p in before}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "split, tasks",
    [
        ("train_tasks", []),
        ("validation_tasks", []),
        ("validation_tasks", [EvaluationTask(id="train", query="leak")]),
        ("test_tasks", [EvaluationTask(id="validation", query="leak")]),
        ("train_tasks", [EvaluationTask(id="duplicate", query="q")] * 2),
        ("train_tasks", [EvaluationTask(id="", query="q")]),
    ],
)
async def test_invalid_splits_fail_before_creating_run(kwargs, split, tasks):
    kwargs[split] = tasks
    with pytest.raises(ValueError):
        await evolve(**kwargs)
    assert not kwargs["output"].exists()


@pytest.mark.asyncio
async def test_context_tail_and_bounded_rejections_do_not_leak(kwargs):
    contexts = []
    memories = []

    async def refiner(graph, context, rejections):
        contexts.append(context)
        memories.append(rejections)
        return {"unknown": "malformed"}

    kwargs.update(
        refiner=refiner, rounds=4, max_context_chars=93, rejection_memory_limit=2
    )
    await evolve(**kwargs)
    for index, context in enumerate(contexts, 1):
        record = read_json(kwargs["output"] / f"rounds/{index:06d}.json")
        full_context = "".join(
            json.dumps(t, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2)
            + "\n"
            for t in record["train_traces"]
        )
        assert context == full_context[-93:]
    assert [len(memory) for memory in memories] == [0, 1, 2, 2]
    assert [r["round"] for r in memories[-1]] == [2, 3]
    assert "SECRET" not in json.dumps([contexts, memories])
    assert len(list((kwargs["output"] / "rejections").glob("*.json"))) == 4


@pytest.mark.asyncio
async def test_injected_token_codec_gets_actual_token_tail(kwargs):
    class Codec:
        def __init__(self):
            self.texts = []
            self.decoded = []

        def tokenize(self, text):
            self.texts.append(text)
            return list(range(len(text.split())))

        def decode(self, tokens):
            self.decoded.append(tokens)
            return " ".join(self.texts[-1].split()[index] for index in tokens)

    codec = Codec()
    contexts = []

    async def refiner(graph, context, rejections):
        contexts.append(context)
        return GraphEdits()

    kwargs.update(refiner=refiner, token_codec=codec, max_context_tokens=7, rounds=2)
    await evolve(**kwargs)
    assert len(codec.texts) == 2
    for text, tokens, context in zip(codec.texts, codec.decoded, contexts):
        assert tokens == list(range(len(text.split())))[-7:]
        assert context == " ".join(text.split()[-7:])


@pytest.mark.asyncio
async def test_token_limit_requires_codec(kwargs):
    kwargs["max_context_tokens"] = 100
    with pytest.raises(ValueError, match="token_codec"):
        await evolve(**kwargs)
    assert not kwargs["output"].exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["error", "wrong_id", "nan"])
async def test_validation_failure_preserves_baseline_and_hides_error(kwargs, failure):
    calls = 0
    memories = []

    async def runner(graph, task):
        nonlocal calls
        if task.id == "validation":
            calls += 1
            if calls > 1:
                if failure == "error":
                    raise OSError("SECRET validation sample")
                if failure == "wrong_id":
                    return ScoredRollout(task_id="SECRET wrong task", score=1)
                return {"task_id": task.id, "score": float("nan")}
        return ScoredRollout(task_id=task.id, score=0.5)

    async def refiner(graph, context, rejections):
        memories.append(rejections)
        return GraphEdits()

    kwargs.update(runner=runner, refiner=refiner, rounds=2)
    await evolve(**kwargs)
    pointer = read_json(kwargs["output"] / "retained.json")
    assert pointer["round"] == 0
    assert pointer["validation_score"] == 0.5
    record = read_json(kwargs["output"] / "rejections/000001.json")
    assert record["validation"]["status"] == "error"
    assert "score" not in record["validation"]
    assert "SECRET" not in json.dumps(memories)
    assert memories[1][0]["candidate_score"] is None


@pytest.mark.asyncio
async def test_initial_validation_error_logs_and_aborts(kwargs):
    async def runner(graph, task):
        raise OSError("evaluator unavailable")

    kwargs["runner"] = runner
    with pytest.raises(RuntimeError, match="Initial validation failed"):
        await evolve(**kwargs)
    assert read_json(kwargs["output"] / "baseline.json")["status"] == "error"
    assert read_json(kwargs["output"] / "retained.json")["validation_score"] is None
    assert not list((kwargs["output"] / "rounds").iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["refiner", "training"])
async def test_adapter_failures_are_rejections(kwargs, failure):
    async def runner(graph, task):
        if task.id == "train":
            return ScoredRollout(task_id="wrong", score=1)
        return ScoredRollout(task_id=task.id, score=0.5)

    async def refiner(graph, context, rejections):
        raise RuntimeError("refiner unavailable")

    kwargs["runner" if failure == "training" else "refiner"] = (
        runner if failure == "training" else refiner
    )
    await evolve(**kwargs)
    record = read_json(kwargs["output"] / "rejections/000001.json")
    assert record["reason"] == f"{failure}_error"
    assert record["validation"]["status"] == "not_run"


@pytest.mark.asyncio
async def test_batches_cycle_sequentially_and_test_is_never_run(kwargs):
    calls = []

    async def runner(graph, task):
        calls.append(task.id)
        return ScoredRollout(task_id=task.id, score=0.5)

    kwargs.update(
        runner=runner,
        train_tasks=[EvaluationTask(id=str(i), query="q") for i in range(3)],
        batch_size=2,
        rounds=3,
    )
    await evolve(**kwargs)
    assert calls == [
        "validation",
        "0",
        "1",
        "validation",
        "2",
        "validation",
        "0",
        "1",
        "validation",
    ]


def test_cli_trusted_factory(kwargs, monkeypatch, capsys):
    adapter = ModuleType("trusted_test_adapter")
    adapter.factory = lambda: kwargs
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    main(
        [
            "--adapter",
            "trusted_test_adapter:factory",
            "--output",
            str(kwargs["output"]),
            "--rounds",
            "0",
        ]
    )
    assert "Offline retained graph:" in capsys.readouterr().out
    assert read_json(kwargs["output"] / "manifest.json")["config"]["rounds"] == 0


def test_cli_rejects_remote_adapter(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--adapter",
                "https://example.com:factory",
                "--output",
                str(tmp_path / "run"),
            ]
        )
    assert error.value.code == 2


@pytest.mark.asyncio
async def test_changed_candidates_keep_last_accepted_graph_and_score(kwargs):
    scores = iter([0.5, 0.8, 0.7, 0.8])
    parents = []
    candidates = []
    memories = []

    async def runner(graph, task):
        score = next(scores) if task.id == "validation" else 0.2
        return ScoredRollout(task_id=task.id, score=score)

    async def refiner(graph, context, rejections):
        parents.append(graph)
        memories.append(rejections)
        edits = GraphEdits.model_validate(
            {
                "delete_edges": [{"source": "Start", "target": "End"}],
                "add_edges": [
                    {
                        "source": "Start",
                        "target": "End",
                        "guidance": f"revision {len(parents)}",
                    }
                ],
            }
        )
        candidates.append(prepare_candidate(graph, edits))
        return edits

    kwargs.update(runner=runner, refiner=refiner, rounds=3)
    result = await evolve(**kwargs)
    assert parents == [kwargs["initial_graph"], candidates[0], candidates[0]]
    assert result == candidates[2]
    output = kwargs["output"]
    rejected = read_json(output / "rejections/000002.json")
    assert rejected["retained_score"] == 0.8
    assert rejected["candidate"] == candidates[1].model_dump(mode="json")
    assert memories[-1] == [
        {
            "round": 2,
            "reason": "validation_worse",
            "retained_score": 0.8,
            "candidate_score": 0.7,
            "edits": rejected["edits"],
            "candidate_fingerprint": candidates[1].fingerprint,
        }
    ]
    pointer = read_json(output / "retained.json")
    assert pointer["fingerprint"] == result.fingerprint
    assert pointer["validation_score"] == 0.8
    assert len(list((output / "graphs").iterdir())) == 4
    assert (
        ProceduralGraph.model_validate(read_json(output / pointer["graph"])).fingerprint
        == result.fingerprint
    )


@pytest.mark.asyncio
async def test_unknown_tool_rejected_without_candidate_validation(kwargs):
    async def refiner(graph, context, rejections):
        return GraphEdits.model_validate(
            {
                "add_nodes": [{"id": "missing_tool", "type": "ACTION"}],
            }
        )

    kwargs["refiner"] = refiner
    await evolve(**kwargs)
    record = read_json(kwargs["output"] / "rejections/000001.json")
    assert record["reason"] == "proposal_error"
    assert record["validation"]["status"] == "not_run"
    assert "Unknown ACTION" in record["error"]


@pytest.mark.asyncio
async def test_reordered_graph_artifact_is_deduplicated(kwargs):
    async def refiner(graph, context, rejections):
        return GraphEdits.model_validate(
            {
                "delete_nodes": [node.id for node in graph.nodes],
                "add_nodes": [node.model_dump() for node in reversed(graph.nodes)],
                "add_edges": [edge.model_dump() for edge in graph.edges],
            }
        )

    kwargs["refiner"] = refiner
    result = await evolve(**kwargs)
    assert result.fingerprint == kwargs["initial_graph"].fingerprint
    assert len(list((kwargs["output"] / "graphs").iterdir())) == 1
    assert read_json(kwargs["output"] / "rounds/000001.json")["status"] == "accepted"


@pytest.mark.asyncio
async def test_zero_memory_still_persists_rejections(kwargs):
    async def refiner(graph, context, rejections):
        assert rejections == []
        return {"invalid": True}

    kwargs.update(refiner=refiner, rejection_memory_limit=0, rounds=2)
    await evolve(**kwargs)
    assert len(list((kwargs["output"] / "rejections").iterdir())) == 2


@pytest.mark.asyncio
async def test_different_start_cannot_silently_resume_existing_run(kwargs):
    await evolve(**kwargs)
    kwargs["initial_graph"] = prepare_candidate(
        kwargs["initial_graph"],
        GraphEdits.model_validate({"add_nodes": [{"id": "another", "type": "STATUS"}]}),
    )
    with pytest.raises(FileExistsError):
        await evolve(**kwargs)


@pytest.mark.asyncio
async def test_pointer_replacement_is_atomic(kwargs, monkeypatch):
    from hyperforge.procedural import evolution

    replace = evolution.os.replace
    replacements = []

    def observe(source, target):
        assert source.parent == target.parent == kwargs["output"]
        replacement = read_json(source)
        assert (kwargs["output"] / replacement["graph"]).exists()
        replacements.append(replacement)
        replace(source, target)

    monkeypatch.setattr(evolution.os, "replace", observe)
    await evolve(**kwargs)
    assert [value["validation_score"] for value in replacements] == [None, 0.5, 0.5]


@pytest.mark.asyncio
async def test_partial_validation_is_not_an_aggregate_score(kwargs):
    validation_calls = 0

    async def runner(graph, task):
        nonlocal validation_calls
        if task.id.startswith("validation"):
            validation_calls += 1
            if validation_calls == 4:
                raise RuntimeError("second validation rollout failed")
        return ScoredRollout(task_id=task.id, score=1)

    kwargs["validation_tasks"].append(EvaluationTask(id="validation2", query="secret"))
    kwargs["runner"] = runner
    await evolve(**kwargs)
    record = read_json(kwargs["output"] / "rejections/000001.json")
    assert len(record["validation"]["traces"]) == 1
    assert "score" not in record["validation"]
    assert read_json(kwargs["output"] / "retained.json")["round"] == 0


@pytest.mark.asyncio
async def test_task_mutations_do_not_escape_adapter(kwargs):
    task = kwargs["train_tasks"][0]
    task.metadata["nested"] = []

    async def runner(graph, task):
        if task.id == "train":
            assert task.metadata["nested"] == []
            task.metadata["nested"].append("adapter-owned")
        return ScoredRollout(task_id=task.id, score=0.5)

    kwargs.update(runner=runner, rounds=2)
    await evolve(**kwargs)
    assert task.metadata == {"nested": []}


@pytest.mark.asyncio
async def test_missing_provenance_does_not_claim_output(kwargs):
    kwargs["provenance"] = {"model": "fake"}
    with pytest.raises(ValueError, match="provenance"):
        await evolve(**kwargs)
    assert not kwargs["output"].exists()


@pytest.mark.asyncio
async def test_only_current_batch_queries_without_gold_enter_context(kwargs):
    contexts = []

    async def runner(graph, task):
        return ScoredRollout(task_id=task.id, score=0.5)

    async def refiner(graph, context, rejections):
        contexts.append(context)
        return GraphEdits()

    kwargs.update(
        runner=runner,
        refiner=refiner,
        rounds=2,
        train_tasks=[
            EvaluationTask(id=str(i), query=f"objective {i}", expected="GOLD SECRET")
            for i in range(2)
        ],
    )
    await evolve(**kwargs)
    for index, context in enumerate(contexts):
        assert json.loads(context)["query"] == f"objective {index}"
        assert f"objective {1 - index}" not in context
        assert "GOLD SECRET" not in context
        record = read_json(kwargs["output"] / f"rounds/{index + 1:06d}.json")
        assert record["train_traces"][0]["query"] == f"objective {index}"
        assert "expected" not in record["train_traces"][0]


@pytest.mark.asyncio
async def test_negative_memory_includes_edits_and_only_structural_diagnostics(kwargs):
    memories = []

    async def refiner(graph, context, rejections):
        memories.append(rejections)
        return {"delete_nodes": ["missing"]}

    kwargs.update(refiner=refiner, rounds=2)
    await evolve(**kwargs)
    summary = memories[1][0]
    assert summary["edits"] == {"delete_nodes": ["missing"]}
    assert summary["candidate_fingerprint"] is None
    assert "Unknown node deletion" in summary["structural_diagnostic"]


@pytest.mark.asyncio
async def test_rejection_character_budget_skips_oversized_keeps_newest_complete(kwargs):
    memories = []

    async def refiner(graph, context, rejections):
        memories.append(rejections)
        return {"delete_nodes": ["missing" if len(memories) != 2 else "x" * 4000]}

    kwargs.update(refiner=refiner, rounds=5, max_rejection_chars=800)
    await evolve(**kwargs)
    for memory in memories:
        assert (
            len(
                json.dumps(
                    memory, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2
                )
            )
            <= 800
        )
        for summary in memory:
            assert summary["edits"] == {"delete_nodes": ["missing"]}
            assert "Unknown node deletion" in summary["structural_diagnostic"]
    assert all(item["round"] != 2 for memory in memories for item in memory)
    assert memories[-1][-1]["round"] == 4
    assert len(list((kwargs["output"] / "rejections").iterdir())) == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [0, -1])
async def test_rejection_budget_must_be_positive(kwargs, limit):
    kwargs["max_rejection_chars"] = limit
    with pytest.raises(ValueError, match="max_rejection_chars"):
        await evolve(**kwargs)
    assert not kwargs["output"].exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field, value",
    [
        ("query", "different"),
        ("expected", {"gold": 2}),
        ("metadata", {"source": "new"}),
    ],
)
async def test_split_hashes_distinguish_changed_content_same_ids(kwargs, field, value):
    kwargs["rounds"] = 0
    await evolve(**kwargs)
    first = read_json(kwargs["output"] / "manifest.json")
    kwargs["output"] = kwargs["output"].with_name("changed")
    kwargs["train_tasks"] = [kwargs["train_tasks"][0].model_copy(update={field: value})]
    await evolve(**kwargs)
    changed = read_json(kwargs["output"] / "manifest.json")
    assert first["splits"] == changed["splits"]
    assert (
        first["split_content_hashes"]["train"]
        != changed["split_content_hashes"]["train"]
    )
    assert (
        first["split_content_hashes"]["validation"]
        == changed["split_content_hashes"]["validation"]
    )
    assert (
        first["split_content_hashes"]["test"] == changed["split_content_hashes"]["test"]
    )


@pytest.mark.asyncio
async def test_artifacts_are_private_even_with_permissive_umask(kwargs):
    import os

    root = kwargs["output"]
    kwargs["output"] = root / "experiment" / "nested" / "run"
    old = os.umask(0)
    try:
        await evolve(**kwargs)
    finally:
        os.umask(old)
    output = kwargs["output"]
    assert root.stat().st_mode & 0o777 == 0o700
    assert output.parent.stat().st_mode & 0o777 == 0o700
    assert output.parent.parent.stat().st_mode & 0o777 == 0o700
    assert output.stat().st_mode & 0o777 == 0o700
    for path in output.rglob("*"):
        assert path.stat().st_mode & 0o777 == (
            0o700 if path.is_dir() else 0o600 if path.name == "retained.json" else 0o400
        )


@pytest.mark.asyncio
async def test_refiner_cannot_mutate_rejection_history(kwargs):
    calls = 0

    async def refiner(graph, context, rejections):
        nonlocal calls
        calls += 1
        if rejections:
            assert rejections[0]["edits"] == {"delete_nodes": ["missing"]}
            rejections[0]["edits"]["delete_nodes"].append("mutated")
        return {"delete_nodes": ["missing"]}

    kwargs.update(refiner=refiner, rounds=3)
    await evolve(**kwargs)
    assert calls == 3
    assert read_json(kwargs["output"] / "rejections/000001.json")["edits"] == {
        "delete_nodes": ["missing"]
    }


def test_artifact_publication_is_atomic_private_and_no_clobber(tmp_path, monkeypatch):
    from hyperforge.procedural import evolution

    store = evolution._ArtifactStore(tmp_path / "run")
    destination = store.output / "rounds/000001.json"
    link = evolution.os.link
    publications = []

    def observe(source, target):
        assert source.parent == target.parent
        assert source.stat().st_mode & 0o777 == 0o400
        assert read_json(source) == {"complete": True}
        assert not target.exists()
        publications.append(source)
        link(source, target)

    monkeypatch.setattr(evolution.os, "link", observe)
    store.write("rounds/000001.json", {"complete": True})
    assert read_json(destination) == {"complete": True}
    assert publications and not publications[0].exists()
    monkeypatch.setattr(evolution.os, "link", link)
    with pytest.raises(FileExistsError):
        store.write("rounds/000001.json", {"replacement": True})
    assert read_json(destination) == {"complete": True}
    assert list(destination.parent.iterdir()) == [destination]


@pytest.mark.parametrize("failure", ["write", "fsync", "link"])
def test_artifact_write_failure_leaves_no_final_or_temporary(
    tmp_path, monkeypatch, failure
):
    from hyperforge.procedural import evolution

    store = evolution._ArtifactStore(tmp_path / "run")
    destination = store.output / "rounds/000001.json"

    def fail(*args):
        assert not destination.exists()
        raise OSError("injected write failure")

    if failure == "write":
        create = evolution.tempfile.NamedTemporaryFile

        def partial_write(*args, **kwargs):
            stream = create(*args, **kwargs)

            def write(content):
                stream.file.write(content[:3])
                stream.file.flush()
                assert Path(stream.name).stat().st_mode & 0o777 == 0o600
                fail()

            stream.write = write
            return stream

        monkeypatch.setattr(evolution.tempfile, "NamedTemporaryFile", partial_write)
    else:
        monkeypatch.setattr(evolution.os, failure, fail)
    with pytest.raises(OSError, match="injected write failure"):
        store.write("rounds/000001.json", {"complete": True})
    assert not destination.exists()
    assert list(destination.parent.iterdir()) == []
