# hyperforge_a2a

A2A client agent for Hyperforge. It connects to an external Agent2Agent (A2A)
server and exposes streamed text responses as Hyperforge context.

Connection settings belong to an A2A driver. The context module's `source`
references that driver's identifier and keeps only per-workflow routing options.

```yaml
drivers:
  - identifier: remote-a2a
    name: Remote A2A
    provider: a2a
    config:
      endpoint: https://a2a.example.com/api/v1/account/account-id/agent/agent-id/a2a
      ca_certificate: |-
        -----BEGIN CERTIFICATE-----
        ...
        -----END CERTIFICATE-----
      authorization: Bearer service-account-key
      read_timeout_seconds: 120

context:
  - module: a2a
    source: remote-a2a
    remote_workflow_id: default
    valid_headers:
      - authorization
```

The driver's `endpoint` identifies one remote agent connection and contains its
account and agent identifiers when required by the server route. It accepts
either a dedicated gRPC address such as `localhost:8034` or an HTTP(S) base URL
such as `https://a2a.example.com/api/v1/account/account-id/agent/agent-id/a2a`.
For HTTP(S), the client resolves `/.well-known/agent-card.json` and uses the
transport advertised by the remote Agent Card (JSON-RPC, HTTP+JSON, or gRPC).
A direct gRPC address has no URL path, so it must be dedicated to the configured
remote agent. HTTPS secures discovery; `use_tls` independently controls direct
gRPC connections.

For direct gRPC connections, set `use_tls` and optionally configure
`ca_certificate` with PEM content. Servers requiring mTLS also need both
`client_certificate_chain` and `client_private_key` as PEM content. The driver
applies `read_timeout_seconds` to the complete remote interaction, including
any feedback round trips. `authorization` and `client_private_key` are encrypted
at rest and omitted from driver API responses.

The driver can store a static `authorization` value. When `authorization` is in
the context module's `valid_headers`, an incoming `Authorization: Bearer <key>`
overrides that static value for the request. It is attached to transport
metadata and is never serialized into A2A message metadata or workflow
configuration.

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

SaaS deployments must set `A2A_AUTH_ENABLED=true` and
`A2A_AUTHORIZER_URL` to the internal regional authorizer base URL. The server
authorizes task RPCs against
`POST /authorize/api/v1/agent/<agent_id>/a2a`; Agent Card discovery remains
public. This server-side authentication configuration is SaaS-only; standalone
does not expose it. `A2A_AUTHORIZER_TIMEOUT_SECONDS` defaults to 5 seconds, and
authorizer failures deny the RPC.

The server accepts these optional message metadata fields:

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
