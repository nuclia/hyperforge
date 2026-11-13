# Security Review Exclusions

These exclusions prevent repeated false positives. Revisit them when deployment or product
boundaries change.

## Deployment-Layer Controls

- Do not report missing TLS termination or ingress rate limiting in application code without
  evidence that Hyperforge itself is expected to provide those controls.
- Do not report the mere acceptance of `X-STF-*` or `X-NUCLIADB-*` identity headers as spoofing.
  Production relies on an upstream authorizer and network policy to strip and inject them. Mixing
  identities, skipping route authorization, or incorrect resource scoping inside Hyperforge is
  still in scope.

## Standalone Mode

- Standalone mode intentionally grants local callers all application roles and uses account
  `local`. It is not a multi-tenant production authentication boundary.
- Open standalone authentication alone is not a finding. Bypassing explicitly configured MCP JWT
  protection, leaking credentials, or exposing the local mode contrary to its documented trust
  model remains in scope.
- The default wildcard CORS setting in standalone mode is not independently actionable. Report it
  only with a demonstrated credential-bearing cross-origin attack or a changed deployment claim.

## Development And Tests

- Dummy tokens, generated keys, localhost endpoints, and PostgreSQL test credentials confined to
  tests/fixtures are not secrets unless reused outside test scope.
- Mocked HTTP, OAuth, MCP, Redis, PostgreSQL, and NucliaDB behavior is not a production finding by
  itself.
- Local/debug restricted-agent execution is intentionally less isolated. Report a weakness only
  when it violates documented behavior, escapes an enabled production sandbox boundary, or is
  reachable in a production-default configuration.

## Scanner Duplication

- Do not restate Polaris, Black Duck, TruffleHog, or zizmor output without confirming reachability,
  impact, and relevance to Hyperforge.
