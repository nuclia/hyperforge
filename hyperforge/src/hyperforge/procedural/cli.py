"""Explicit local entry point for trusted, executable Python adapters."""

import argparse
import asyncio
import importlib
from collections.abc import Sequence

from .evolution import evolve


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run offline procedural evolution. This does not promote production state.",
        epilog=(
            "SECURITY: --adapter imports and executes trusted local Python code. "
            "Never use an untrusted module. The synchronous zero-argument factory must "
            "return evolve() keyword arguments, including initial_graph, isolated runner, "
            "refiner, train_tasks, validation_tasks, available_tools and provenance "
            "(model/evaluator/tools). No remote adapter loading is supported."
        ),
    )
    parser.add_argument(
        "--adapter",
        required=True,
        metavar="MODULE:FACTORY",
        help="Explicitly trusted Python adapter factory",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="New local run directory; existing paths are rejected",
    )
    parser.add_argument(
        "--rounds", type=int, default=10, help="Evolution rounds (default: 10)"
    )
    args = parser.parse_args(argv)
    module_name, separator, factory_name = args.adapter.partition(":")
    if (
        not separator
        or not all(part.isidentifier() for part in module_name.split("."))
        or not factory_name.isidentifier()
    ):
        parser.error(
            "--adapter must be a local Python module:function, not a URL or file path"
        )
    if args.rounds < 0:
        parser.error("--rounds must be nonnegative")
    factory = getattr(importlib.import_module(module_name), factory_name)
    kwargs = factory()
    if not isinstance(kwargs, dict):
        parser.error("adapter factory must return a dict of evolve() keyword arguments")
    retained = asyncio.run(
        evolve(**{**kwargs, "output": args.output, "rounds": args.rounds})
    )
    print(f"Offline retained graph: {retained.fingerprint}")


if __name__ == "__main__":
    main()
