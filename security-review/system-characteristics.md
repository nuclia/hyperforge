# Hyperforge Security Characteristics

Last reviewed: 2026-08-17

## Architecture

- Python 3.12 `uv` workspace containing the core `hyperforge` package and first-party packages in
  `agents/`.
- Core HTTP surfaces use FastAPI/Starlette and include management, interaction, websocket, MCP, and
  OAuth callback routes.
- The main service persists configuration in PostgreSQL, uses Redis/Valkey for broker and cache
  behavior, and uses NucliaDB for memory/search operations.
- Standalone mode combines the API and runner in one process and can use local in-memory state. It
  deliberately grants all local requests application roles, with optional JWT protection for MCP.
- The Vue frontend is built into the core Python package when present.

## Trust Boundaries

- Production authentication trusts identity and roles injected by an upstream authorizer in
  `X-STF-*` or `X-NUCLIADB-*` headers. Route authorization and resource scoping remain application
  responsibilities.
- Agent, workflow, account, session, question, and OAuth identifiers cross API, broker, cache, and
  persistence boundaries and must remain consistently scoped.
- The HTTP and MCP HTTP agents make outbound requests from configured URLs. `SafeTransport` and DNS
  checks are intended to prevent access to private and non-global addresses.
- The restricted agent executes generated Python through RestrictedPython. Production can place
  execution in a separate process/container reached over a Unix socket; local/debug modes have
  weaker isolation and must not be treated as equivalent security boundaries.
- MCP stdio launches only entries from a source allowlist and passes a reduced environment plus
  explicitly configured values to child processes.
- OAuth routing state for MCP is authenticated and encrypted with Fernet and has a ten-minute TTL.
  Standalone MCP bearer validation supports RSA SHA-2 algorithms and can enforce issuer, audience,
  expiry, not-before, and scopes.

## CI And Delivery

- Package workflows test and publish the core and individual agent packages.
- Container images are built from `HYPERFORGE.Dockerfile` and published to GHCR.
- Polaris, Black Duck, TruffleHog, and zizmor provide deterministic security scanning.
- The security review agent runs without application secrets and verifies behavior with focused
  local tests rather than a production-like service stack.
