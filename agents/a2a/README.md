# hyperforge_a2a

A2A client agent for Hyperforge. It connects to an external Agent2Agent (A2A)
server and exposes streamed text responses as Hyperforge context.

`source` accepts either a gRPC address such as `localhost:8034` or an HTTP(S)
base URL such as `http://localhost:9999`. For HTTP(S) sources, the client
resolves `/.well-known/agent-card.json` and uses the transport advertised by
the remote Agent Card.

## Serving Hyperforge over A2A

The A2A gRPC server represents one configured Hyperforge agent. Set
`A2A_ACCOUNT` and `A2A_AGENT_ID` before starting it; startup fails when either
value is missing or the selected agent has no workflows. Its Agent Card exposes
one text input/output skill per configured workflow. Skill IDs use
`<agent_id>:<workflow_id>`.

The server accepts these optional message metadata fields:

- `account` and `agent_id`: may repeat the configured server identity. A
	different value is rejected.
- `workflow_id`: selects an advertised workflow and defaults to `default`.
	Unknown workflows are rejected.
- `session`: a non-empty session identifier. It defaults to the A2A context
	ID.
- `arguments`: an object with non-empty string keys and scalar values. Values
	are passed to Hyperforge as strings.
- `headers`: an object with non-empty string keys and scalar values. Only
	headers named in `A2A_ALLOWED_FORWARDED_HEADERS` are forwarded; all other
	headers are rejected.

All other metadata fields, empty string values, non-object `headers` or
`arguments`, and nested header or argument values are validation errors. The
server reports these errors as failed A2A tasks before starting a Hyperforge
interaction.
