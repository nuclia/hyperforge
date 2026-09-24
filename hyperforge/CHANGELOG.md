# CHANGELOG

## Unreleased

- Added explicit scoped Code Mode capabilities and child-tool inheritance policy.
- Hardened scoped Code Mode with fail-closed remote execution, bounded JSON
  transport, execution limits, atomic callback cleanup, immutable sandbox
  admission/configuration snapshots, and memory-bounded, key-safe sanitized
  nested-call events. Scoped worker errors are redacted before transport and
  remain redacted if transport fails; scoped source/result prechecks stop at their
  configured bounds. Nested-call and output violations are terminal, and late
  nested events retain the turn that originated the invocation. Event backend,
  global call, projected-result, cumulative-result, worker output serialization,
  and execution-marker failures now fail closed across caught retries. Supported
  marked protocol models remain available in run request local and global values,
  while plain marker-like dictionaries round-trip without retyping. Integer event
  markers and exact projected-result response envelopes are now bounded before
  nested execution or completion emission.

## 1.0.0

- Initial version
