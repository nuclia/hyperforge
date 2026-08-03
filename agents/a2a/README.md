# hyperforge_a2a

A2A client agent for Hyperforge. It connects to an external Agent2Agent (A2A)
server and exposes streamed text responses as Hyperforge context.

`source` accepts either a gRPC address such as `localhost:8034` or an HTTP(S)
base URL such as `http://localhost:9999`. For HTTP(S) sources, the client
resolves `/.well-known/agent-card.json` and uses the transport advertised by
the remote Agent Card (JSON-RPC, HTTP+JSON, or gRPC). HTTPS secures discovery;
`use_tls` independently controls gRPC connections.

For direct gRPC connections, set `use_tls` and optionally configure
`tls_ca_certificate_path`. Servers requiring mTLS also need both
`tls_client_certificate_chain_path` and `tls_client_private_key_path`. The
client applies `read_timeout_seconds` to the complete remote interaction,
including any feedback round trips.

When a remote A2A task enters `input-required`, the client creates a standard
Hyperforge feedback request using the remote question and response schema. The
answer supplied through Hyperforge's normal feedback path resumes the same A2A
task with its original task ID, context ID, and feedback ID. Remote failed or
cancelled task states are reported as A2A client errors.

## Serving Hyperforge over A2A

The A2A server is gRPC-only and represents one configured Hyperforge agent. Set
`A2A_ACCOUNT` and `A2A_AGENT_ID` before starting it; startup fails when either
value is missing or the selected agent has no workflows. Its Agent Card exposes
one text input/output skill per configured workflow. Skill IDs use
`<agent_id>:<workflow_id>`.

Set `A2A_PUBLIC_URL` whenever the server binds a wildcard address or uses TLS;
this is the address published in the Agent Card. TLS requires a certificate
chain and private key; configuring a client CA enables mTLS.

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

## Feedback replies

When a workflow requires user input, the server transitions the A2A task to
`input-required`. Its status message includes the Hyperforge response schema,
request ID, and `feedback_id`. Send the answer as a new A2A user message with
the same `task_id` and `context_id`, and include the returned `feedback_id` in
message metadata. The text content of that message is delivered to the pending
Hyperforge feedback request, after which the original task continues streaming
artifacts until it completes or fails.

Pending task correlation is stored in Redis under
`A2A_TASK_STORE_PREFIX` and expires after `A2A_TASK_TTL_SECONDS` (300 seconds
by default). A missing, expired, or no-longer-active task, a mismatched context,
or an invalid feedback ID is rejected as a failed A2A task. Cancelling a task
removes its pending feedback correlation and later replies are rejected.

Feedback may arrive at any A2A server pod. The receiving pod validates it and
relays it through the shared broker to the pod that owns the live workflow,
which relays the resulting A2A events back to the request. The owner must still
be alive: after its process restarts, the durable task can be queried but its
paused workflow cannot be resumed and the client must resend the request.

`feedback_id` is valid only on a reply. All other metadata fields, empty string values, non-object `headers` or
`arguments`, and nested header or argument values are validation errors. The
server reports these errors as failed A2A tasks before starting a Hyperforge
interaction.
