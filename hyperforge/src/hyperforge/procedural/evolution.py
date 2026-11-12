"""Sequential, offline graph evolution with caller-owned execution and scoring.

The runner must execute actual rollouts in an isolated adapter, not replay traces.
This module neither installs tools into a registry nor promotes production state.
Artifacts contain potentially sensitive traces; choose a suitably private output
directory. The default context bound is characters, NOT an estimate of tokens.
"""

import hashlib
import json
import os
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from .graph import GraphEdits, ProceduralGraph, prepare_candidate


class EvaluationTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    query: str
    expected: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoredRollout(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    score: FiniteFloat = Field(ge=0, le=1)
    trajectory: list[dict[str, Any]] = Field(default_factory=list)
    output: str = ""
    metrics: dict[str, FiniteFloat] = Field(default_factory=dict)


class TokenCodec(Protocol):
    """Supply the refiner model's tokenizer; no tokenizer dependency is assumed."""

    def tokenize(self, text: str) -> Sequence[int]: ...

    def decode(self, tokens: Sequence[int]) -> str: ...


Runner = Callable[[ProceduralGraph, EvaluationTask], Awaitable[ScoredRollout]]
Refiner = Callable[[ProceduralGraph, str, list[dict]], Awaitable[GraphEdits]]


def _json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2
    )


class _ArtifactStore:
    """Exclusive creation prevents accidental resume or experiment overwrites."""

    def __init__(self, output: Path):
        missing_parents = []
        parent = output.parent
        while not parent.exists():
            missing_parents.append(parent)
            parent = parent.parent
        for parent in reversed(missing_parents):
            parent.mkdir(mode=0o700, exist_ok=True)
        output.mkdir(mode=0o700, exist_ok=False)
        self.output = output
        for name in ("graphs", "rounds", "rejections"):
            (output / name).mkdir(mode=0o700)

    def write(self, name: str, value: Any) -> None:
        content = _json(value)
        path = self.output / name
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(content + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                temporary.chmod(0o400)
            # Hard-link publication is atomic and fails if the destination exists.
            os.link(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def graph(self, graph: ProceduralGraph) -> None:
        name = f"graphs/{graph.fingerprint}.json"
        data = graph.model_dump(mode="json")
        data["nodes"].sort(key=lambda node: node["id"])
        data["edges"].sort(
            key=lambda edge: (edge["source"], edge["target"], edge["relation"])
        )
        path = self.output / name
        if path.exists():
            if json.loads(path.read_text()) != data:
                raise ValueError("Graph fingerprint collision")
            return
        self.write(name, data)

    def retain(
        self, graph: ProceduralGraph, score: float | None, round_id: int
    ) -> None:
        value = {
            "fingerprint": graph.fingerprint,
            "graph": f"graphs/{graph.fingerprint}.json",
            "validation_score": score,
            "round": round_id,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=self.output, delete=False
        ) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(_json(value) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            os.replace(temporary, self.output / "retained.json")
        finally:
            temporary.unlink(missing_ok=True)


async def evolve(
    initial_graph: ProceduralGraph,
    *,
    output: str | Path,
    train_tasks: Sequence[EvaluationTask],
    validation_tasks: Sequence[EvaluationTask],
    runner: Runner,
    refiner: Refiner,
    available_tools: Sequence[str],
    provenance: dict[str, Any],
    test_tasks: Sequence[EvaluationTask] = (),
    rounds: int = 10,
    batch_size: int = 1,
    max_context_chars: int = 16000,
    max_context_tokens: int | None = None,
    token_codec: TokenCodec | None = None,
    rejection_memory_limit: int = 10,
    max_rejection_chars: int = 16000,
) -> ProceduralGraph:
    """Return the final locally retained graph, accepting validation ties.

    ``output`` must not exist, even for zero rounds. ``provenance`` must include
    ``model``, ``evaluator`` and ``tools`` (JSON-serializable caller descriptions).
    Training uses consecutive batches, cycling after the last partial batch.
    Test tasks are recorded only, never executed or supplied to the refiner.
    Initial validation is cached; a baseline failure logs and raises RuntimeError.
    Later runner/refiner/proposal failures reject that round and preserve state.
    File I/O failures propagate, rather than pretending an artifact was committed.

    Refiner context is the tail of the CURRENT BATCH's concatenated scored training
    rollouts with task queries (not expected answers), bounded
    by characters unless ``max_context_tokens`` and a token codec are supplied.
    Rejection memory includes edits, candidate fingerprints, aggregate scores and
    proposal structural diagnostics, never validation examples or errors. It is
    bounded by count and the serialized list's character length. Complete newest
    entries take priority; oversized entries are skipped (never truncated). An
    empty list is supplied when none fit, even for a one-character budget.
    Split content hashes cover ordered task JSON, including expected answers and
    metadata; only hashes and IDs are written to the manifest.
    """
    if rounds < 0 or batch_size < 1 or rejection_memory_limit < 0:
        raise ValueError(
            "rounds/memory limit must be nonnegative; batch_size must be positive"
        )
    if max_context_chars < 1:
        raise ValueError("max_context_chars must be positive")
    if max_rejection_chars < 1:
        raise ValueError("max_rejection_chars must be positive")
    if max_context_tokens is not None and (
        max_context_tokens < 1 or token_codec is None
    ):
        raise ValueError("max_context_tokens requires a positive limit and token_codec")
    if not {"model", "evaluator", "tools"} <= provenance.keys():
        raise ValueError("provenance requires model, evaluator and tools")
    splits = {
        "train": [EvaluationTask.model_validate(t.model_dump()) for t in train_tasks],
        "validation": [
            EvaluationTask.model_validate(t.model_dump()) for t in validation_tasks
        ],
        "test": [EvaluationTask.model_validate(t.model_dump()) for t in test_tasks],
    }
    if not splits["train"] or not splits["validation"]:
        raise ValueError("train and validation splits must be nonempty")
    ids = [task.id for tasks in splits.values() for task in tasks]
    if any(not task_id.strip() for task_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("Task IDs must be nonempty, unique and disjoint across splits")
    retained = ProceduralGraph.model_validate(initial_graph.model_dump(mode="json"))
    tools = tuple(available_tools)
    retained.check_tools(tools)
    manifest = {
        "schema_version": 1,
        "initial_fingerprint": retained.fingerprint,
        "splits": {name: [t.id for t in tasks] for name, tasks in splits.items()},
        "split_content_hashes": {
            name: hashlib.sha256(
                _json([task.model_dump(mode="json") for task in tasks]).encode("utf-8")
            ).hexdigest()
            for name, tasks in splits.items()
        },
        "provenance": provenance,
        "available_tools": tools,
        "config": {
            "rounds": rounds,
            "batch_size": batch_size,
            "max_context_chars": max_context_chars
            if max_context_tokens is None
            else None,
            "max_context_tokens": max_context_tokens,
            "rejection_memory_limit": rejection_memory_limit,
            "max_rejection_chars": max_rejection_chars,
        },
    }
    _json(
        manifest
    )  # Fail before claiming the output directory if provenance is invalid.
    store = _ArtifactStore(Path(output))
    store.write("manifest.json", manifest)
    store.graph(retained)
    store.retain(retained, None, 0)

    async def evaluate(
        graph: ProceduralGraph, tasks: Sequence[EvaluationTask], traces: list
    ) -> float:
        for task in tasks:
            # Copies keep mutable nested model values under adapter ownership.
            result = await runner(
                graph.model_copy(deep=True), task.model_copy(deep=True)
            )
            rollout = ScoredRollout.model_validate(
                result.model_dump() if isinstance(result, BaseModel) else result
            )
            if rollout.task_id != task.id:
                raise ValueError(
                    f"Runner task ID mismatch: expected {task.id!r}, got {rollout.task_id!r}"
                )
            trace = rollout.model_dump(mode="json")
            trace["query"] = task.query
            _json(trace)
            traces.append(trace)
        return sum(trace["score"] for trace in traces) / len(traces)

    baseline: dict[str, Any] = {"graph": retained.fingerprint, "traces": []}
    try:
        retained_score = await evaluate(
            retained, splits["validation"], baseline["traces"]
        )
    except Exception as exc:
        baseline.update(status="error", error=f"{type(exc).__name__}: {exc}")
        store.write("baseline.json", baseline)
        raise RuntimeError(
            "Initial validation failed; initial graph retained without a score"
        ) from exc
    baseline.update(status="scored", score=retained_score)
    store.write("baseline.json", baseline)
    store.retain(retained, retained_score, 0)

    rejections: list[dict[str, Any]] = []
    cursor = 0
    for round_id in range(1, rounds + 1):
        batch = splits["train"][cursor : cursor + batch_size]
        cursor = (cursor + len(batch)) % len(splits["train"])
        record: dict[str, Any] = {
            "round": round_id,
            "parent": retained.fingerprint,
            "retained_score": retained_score,
            "batch_ids": [task.id for task in batch],
            "candidate": None,
            "edits": None,
            "train_traces": [],
            "validation": {"status": "not_run", "traces": []},
        }
        phase = "training"
        candidate = None
        candidate_score = None
        try:
            await evaluate(retained, batch, record["train_traces"])
            training_context = "".join(
                _json(trace) + "\n" for trace in record["train_traces"]
            )
            if max_context_tokens is not None:
                assert token_codec is not None
                context = token_codec.decode(
                    token_codec.tokenize(training_context)[-max_context_tokens:]
                )
            else:
                context = training_context[-max_context_chars:]
            summaries: list[dict[str, Any]] = []
            for rejected in reversed(
                rejections[-rejection_memory_limit:] if rejection_memory_limit else []
            ):
                summary = {
                    "round": rejected["round"],
                    "reason": rejected["reason"],
                    "retained_score": rejected["retained_score"],
                    "candidate_score": rejected["validation"].get("score"),
                    "edits": rejected["edits"],
                    "candidate_fingerprint": rejected.get("candidate_fingerprint"),
                }
                if rejected["reason"] == "proposal_error":
                    summary["structural_diagnostic"] = rejected["error"]
                bounded = [summary, *summaries]
                if len(_json(bounded)) <= max_rejection_chars:
                    summaries = bounded
            phase = "refiner"
            proposal = await refiner(
                retained.model_copy(deep=True), context, json.loads(_json(summaries))
            )
            phase = "proposal"
            # Preserve even malformed proposals in the audit record.
            try:
                record["edits"] = (
                    proposal.model_dump(mode="json")
                    if isinstance(proposal, BaseModel)
                    else proposal
                )
                _json(record["edits"])
            except (TypeError, ValueError):
                record["edits"] = repr(proposal)
            edits = GraphEdits.model_validate(
                proposal.model_dump() if isinstance(proposal, BaseModel) else proposal
            )
            candidate = prepare_candidate(retained, edits, available_tools=tools)
        except Exception as exc:
            record.update(
                status="rejected",
                reason=f"{phase}_error",
                error=f"{type(exc).__name__}: {exc}",
            )
        else:
            record["candidate"] = candidate.model_dump(mode="json")
            record["candidate_fingerprint"] = candidate.fingerprint
            store.graph(candidate)
            validation: dict[str, Any] = record["validation"]
            try:
                candidate_score = await evaluate(
                    candidate, splits["validation"], validation["traces"]
                )
            except Exception as exc:
                validation.update(status="error", error=f"{type(exc).__name__}: {exc}")
                record.update(status="rejected", reason="validation_error")
            else:
                validation.update(status="scored", score=candidate_score)
                if candidate_score >= retained_score:
                    record.update(status="accepted", reason="validation_not_worse")
                else:
                    record.update(status="rejected", reason="validation_worse")

        store.write(f"rounds/{round_id:06d}.json", record)
        if record["status"] == "accepted":
            assert candidate is not None and candidate_score is not None
            store.retain(candidate, candidate_score, round_id)
            retained, retained_score = candidate, candidate_score
        else:
            store.write(f"rejections/{round_id:06d}.json", record)
            rejections.append(record)
    return retained
