# Security Assessment Report

**Project:** Hyperforge  
**Assessment date:** 2026-08-26  
**Revision reviewed:** `4c0d323606216447f795b7a463657e6d1dcf17e0` (`main`) plus the existing uncommitted changes listed under Scope  
**Assessment type:** Static source, configuration, dependency, secret-pattern, and targeted test review

> **Remediation update (2026-08-26):** All confirmed findings in this report have been addressed in the working tree. HF-01 through HF-09 are fixed pending deployment and release. The validation section below records the original scan; post-remediation checks include a clean npm audit, successful frontend build, lint/type checks, and passing targeted security and agent tests. Docker-dependent integration and image scans remain blocked by the unavailable local Docker daemon.

## Executive Summary

Hyperforge has two credential-disclosure paths that should be addressed before either affected service is reachable from an untrusted network. Standalone mode deliberately grants every request all roles while exposing endpoints that return and replace the complete agent configuration. The full API also mounts an unauthenticated internal inspection endpoint that returns decrypted driver configurations for a caller-selected account and agent.

The outbound HTTP safeguard is a useful control, but it validates only IPv4 and performs DNS validation separately from connection establishment. This leaves IPv6 and DNS-rebinding SSRF paths, particularly serious when combined with standalone configuration access. Several request and response paths also buffer data without size limits, creating denial-of-service risk.

The frontend lockfile contains six npm advisories (two high, one moderate, and three low). The two high advisories are not directly exploitable through the application behavior observed: Hyperforge uses only UUID v4, and PostCSS runs as a build dependency without processing user-submitted CSS. They should still be upgraded to restore a clean supply chain.

No obvious committed production credential, private key, SQL injection, shell injection, unsafe YAML deserialization, or direct template XSS was confirmed in the current tracked tree.

### Finding Summary

| ID | Severity | Finding | Status |
|---|---|---|---|
| HF-01 | Critical | Standalone configuration and secrets are exposed without authentication | Remediated |
| HF-02 | High | Internal inspection endpoint returns decrypted credentials without authorization | Remediated |
| HF-03 | High | SSRF protection is bypassable through IPv6 or DNS rebinding | Remediated |
| HF-04 | High | Docker build can copy local secrets and Git metadata into the runtime image | Remediated |
| HF-05 | Medium | MCP request and response buffering permits memory exhaustion | Remediated |
| HF-06 | Medium | HTTP agents accept unbounded remote response bodies | Remediated |
| HF-07 | Medium | Runtime container executes as root | Remediated |
| HF-08 | Medium | Locked npm dependencies have known vulnerabilities | Remediated |
| HF-09 | Low | MCP client certificate temporary files are not deleted | Remediated |

## Immediate Actions

1. Prevent network access to standalone mode and `/api/internal/*` until HF-01 and HF-02 are fixed.
2. Rotate credentials if either endpoint has been exposed to an untrusted network; access logs alone cannot prove secrets were not read.
3. Require authenticated administrative authorization for standalone configuration access and redact every secret-bearing field.
4. Remove or strongly authenticate the internal inspection endpoint and never serialize decrypted driver secrets.
5. Treat configurable outbound URLs as SSRF-capable until DNS and connection validation are made atomic for IPv4 and IPv6.

## Detailed Findings

### HF-01: Standalone Configuration and Secrets Exposed Without Authentication

**Severity:** Critical  
**CWE:** CWE-306 (Missing Authentication), CWE-200 (Exposure of Sensitive Information)  
**Affected component:** Standalone API/UI

**Evidence**

- `hyperforge/src/hyperforge/standalone/settings.py:34-35` binds standalone mode to `0.0.0.0:8080` by default.
- `hyperforge/src/hyperforge/standalone/app.py:91-149` grants every request all application roles. Optional bearer validation applies only to recognized MCP paths.
- `hyperforge/src/hyperforge/standalone/app.py:237-242` mounts the UI router under the open authentication backend.
- `hyperforge/src/hyperforge/standalone/ui_router.py:127-134` returns the complete in-memory configuration using `model_dump()`.
- `hyperforge/src/hyperforge/standalone/ui_router.py:137-177` accepts, activates, and persists a replacement configuration without an authorization check.
- `hyperforge/src/hyperforge/standalone/settings.py:120-126` allows every CORS origin by default, increasing browser-based reachability where the service is network-accessible.

**Impact**

An unauthenticated client can retrieve configured API keys, OAuth client secrets, authorization headers, certificates, prompts, and workflow definitions. The client can replace live behavior, persist malicious configuration, consume paid APIs, and configure outbound requests for data exfiltration or SSRF. Because the full configuration is serialized directly, the database API's encrypted-field redaction is not applied.

**Attack path**

1. Request `GET /api/v1/ui/config` from a reachable standalone instance.
2. Extract credentials and internal endpoint configuration from the response.
3. Submit a modified document to `PUT /api/v1/ui/config` to change active workflows or outbound integrations.

**Remediation**

- Bind standalone mode to loopback by default.
- Require strong authentication and a distinct administrator permission for every `/api/v1/ui/*` endpoint that reads or mutates configuration.
- Disable configuration mutation by default in production.
- Define sensitive fields centrally and redact them from all responses, including custom headers, certificates, MCP environment values, and driver-specific credentials.
- Restrict CORS to explicit trusted origins. Do not combine wildcard origins with credentialed requests.
- Add audit logging for configuration reads and writes without logging configuration values.

### HF-02: Internal Inspection Endpoint Returns Decrypted Credentials

**Severity:** High  
**CWE:** CWE-862 (Missing Authorization), CWE-200 (Exposure of Sensitive Information)  
**Affected component:** Full FastAPI service

**Evidence**

- `hyperforge/src/hyperforge/api/internal/inspect.py:13-29` exposes `GET /api/internal/v1/agent/{kbid}` with a caller-controlled `account` query parameter and no authorization decorator.
- `hyperforge/src/hyperforge/api/app.py:77-83` mounts the internal router in the same application.
- `hyperforge/src/hyperforge/db/agents.py:451-480` decrypts marked fields before returning each driver.
- `hyperforge/src/hyperforge/api/models.py:79-84` includes the full `DriverConfig` list in the response.
- The normal management endpoint uses `dump_without_encrypted_fields()` at `hyperforge/src/hyperforge/api/v1/agents.py:160`; the inspection route does not.

**Impact**

Any caller that can reach the endpoint and identify or enumerate an account and agent can obtain decrypted driver credentials. The account value is supplied by the caller rather than derived from an authenticated identity.

**Remediation**

- Prefer removing the endpoint from the public application and exposing equivalent diagnostics through an authenticated internal service boundary.
- If retained, require service authentication and account-scoped authorization.
- Return only an allowlisted diagnostic projection and always remove encrypted fields after decryption.
- Add regression tests proving anonymous, cross-account, and ordinary member requests cannot access the route.

### HF-03: SSRF Protection Bypass Through IPv6 or DNS Rebinding

**Severity:** High  
**CWE:** CWE-918 (Server-Side Request Forgery)  
**Affected component:** Shared outbound HTTP transport and configurable URL consumers

**Evidence**

- `hyperforge/src/hyperforge/utils/http.py:17-30` resolves only `socket.AF_INET`, so AAAA records are not inspected.
- `hyperforge/src/hyperforge/utils/http.py:32-41` validates DNS before delegating to `httpx`, which resolves the hostname again when connecting.
- `agents/http/src/hyperforge_http/agent.py:36-68`, `agents/external/src/hyperforge_external/agent.py:64-117`, and `agents/mcp/src/hyperforge_mcp/http.py:454-490` consume configurable URLs through this transport.
- `agents/mcp/src/hyperforge_mcp/http.py:454-490` follows redirects; every redirect is checked by a new transport request, but each check still has the resolution race.
- `hyperforge/src/hyperforge/standalone/oauth.py:30-48` retrieves a configured JWKS URL without the shared SSRF transport.

**Impact**

A hostname can pass the IPv4 check while resolving to loopback or private IPv6 for the actual connection. DNS rebinding can also return a public address during validation and an internal address during connection. An attacker with configuration control can target cloud metadata, loopback services, cluster APIs, or internal data stores. HF-01 makes this attack unauthenticated in reachable standalone deployments.

**Remediation**

- Resolve all A and AAAA records and reject every address that is not explicitly permitted and globally routable.
- Connect to a validated address without a second uncontrolled DNS lookup while preserving the original hostname for TLS SNI and certificate validation.
- Revalidate redirect targets and restrict URL schemes to `https` or an explicit allowlist.
- Disable environment-proxy inheritance unless intentionally configured.
- Apply the same transport and policy to JWKS, OAuth metadata, and every configurable outbound endpoint.
- Consider destination allowlists for production rather than relying only on address classification.

### HF-04: Docker Build Can Embed Local Secrets and Git Metadata

**Severity:** High  
**CWE:** CWE-200 (Exposure of Sensitive Information)  
**Affected component:** `HYPERFORGE.Dockerfile`

**Evidence**

- No `.dockerignore` is present.
- `HYPERFORGE.Dockerfile:8` executes `COPY . /app/.`.
- `HYPERFORGE.Dockerfile:15` copies all of `/app` into the final image despite the comment stating that only the virtual environment is copied.

**Impact**

Docker sends the whole local build context unless excluded. Ignored `.env` files, credentials, `.git` history, test cassettes, caches, and developer artifacts can therefore be copied into image layers and the final image. Removing a file in a later layer would not reliably remove it from earlier layers.

**Remediation**

- Add a restrictive `.dockerignore` covering `.git`, environment files, credentials, caches, tests, local virtual environments, build outputs, and editor files.
- Copy manifests first, install dependencies, then copy only required source and built frontend assets.
- Copy the actual virtual environment and required runtime files into the final stage rather than all `/app`.
- Build and scan the resulting image, then inspect its filesystem and layer history before release.

### HF-05: Unbounded MCP Request and Response Buffering

**Severity:** Medium  
**CWE:** CWE-400 (Uncontrolled Resource Consumption)  
**Affected component:** Streamable MCP HTTP endpoint

**Evidence**

- `hyperforge/src/hyperforge/api/v1/mcp_interaction.py:382-392` accumulates all response chunks in a list.
- `hyperforge/src/hyperforge/api/v1/mcp_interaction.py:394-407` reads the complete request body into memory.
- `hyperforge/src/hyperforge/api/v1/mcp_interaction.py:425-428` joins all chunks, requiring another contiguous allocation.
- Standalone mode makes the route unauthenticated unless MCP auth is configured for the exact recognized path.

**Impact**

Large request bodies or generated MCP responses can exhaust process memory. Concurrent requests amplify the effect and can terminate the service.

**Remediation**

- Reject oversized requests at the reverse proxy and application boundary before calling `request.body()`.
- Stream requests and responses using the supported MCP session manager where possible.
- Cap total response bytes, chunk count, request duration, and concurrent MCP operations.
- Add load tests covering compressed and chunked bodies.

### HF-06: HTTP Agents Accept Unbounded Remote Responses

**Severity:** Medium  
**CWE:** CWE-400 (Uncontrolled Resource Consumption)  
**Affected component:** HTTP and external agents

**Evidence**

- `agents/http/src/hyperforge_http/agent.py:38-68` downloads and decodes the complete response.
- `agents/external/src/hyperforge_external/agent.py:118-131` decodes, logs, and stores the complete response.
- `agents/mcp/src/hyperforge_mcp/http.py:459-463` permits a 200-second default MCP timeout.

**Impact**

A malicious or compromised remote server can return an arbitrarily large or slowly streamed body, consuming worker memory and time. External-agent responses may also create very large logs and expose remote response data to centralized logging.

**Remediation**

- Stream responses and stop after a configurable maximum compressed and decompressed byte count.
- Apply connect, read, write, pool, and total operation timeouts.
- Truncate error bodies and avoid logging complete response bodies or model context.

### HF-07: Runtime Container Executes as Root

**Severity:** Medium  
**CWE:** CWE-250 (Execution with Unnecessary Privileges)  
**Affected component:** `HYPERFORGE.Dockerfile`

**Evidence**

- `HYPERFORGE.Dockerfile:14-22` does not create or select a non-root runtime user.
- The image also has no explicit `ENTRYPOINT`, `CMD`, or `HEALTHCHECK`.
- Base image `python:3.14`, bootstrap `uv`, and apt-installed npm are not digest/version pinned at `HYPERFORGE.Dockerfile:1-3,14`.

**Impact**

Application compromise provides root privileges inside the container and increases the consequences of mounted files, permissive runtime settings, or container-engine weaknesses. Mutable inputs reduce build reproducibility.

**Remediation**

- Create a dedicated UID/GID, ensure runtime files are owned appropriately, and set `USER` in the final stage.
- Use a minimal pinned base image and pin bootstrap tooling.
- Set an explicit startup command and health check or document that the orchestrator supplies them.
- Run with a read-only root filesystem, dropped Linux capabilities, no privilege escalation, resource limits, and a restrictive seccomp/AppArmor profile.

### HF-08: Locked npm Dependencies Have Known Vulnerabilities

**Severity:** Medium application risk; upstream advisories include High  
**CWE:** CWE-1104 (Use of Unmaintained Third-Party Components)  
**Affected component:** `hyperforge/frontend/package-lock.json`

**Evidence**

`npm audit --json` reported two high, one moderate, and three low vulnerabilities:

| Package | Locked version | Advisory severity | Reachability assessment |
|---|---:|---:|---|
| `uuid` | 9.0.1 | High | Vulnerability affects v3/v5/v6 buffer APIs; project imports only v4 at `frontend/src/stores/chat.ts:3` and `workflow.ts:3` |
| `postcss` | 8.5.15 | High | Build-time transitive dependency; no user-controlled CSS processing found |
| `nanoid` | 3.3.12 | Moderate | Transitive build dependency; vulnerable custom size APIs are not called by project code |
| `esbuild` | 0.27.7 | Low | Development server issue affecting Windows |
| `vite` | 7.3.5 | Low | Inherits the esbuild advisory |
| `@vitejs/plugin-vue` | lockfile-resolved version | Low | Inherits the Vite advisory |

`npm audit --omit=dev --json` still reported five advisories because the lockfile classifies much of the frontend graph as production dependencies. The production Python service serves compiled static assets and does not execute these npm packages at runtime.

**Remediation**

- Upgrade PostCSS/nanoid to patched versions through the Vite dependency graph.
- Upgrade `uuid` to a patched major version and run frontend tests/build checks.
- Keep Vite and plugin-vue aligned on supported versions.
- Add npm audit/SCA coverage to CI; the Black Duck workflow is configured for uv and its trigger does not match `package-lock.json`.

### HF-09: MCP Client Certificate Temporary Files Are Not Deleted

**Severity:** Low  
**CWE:** CWE-459 (Incomplete Cleanup)  
**Affected component:** MCP HTTP client

**Evidence**

- `agents/mcp/src/hyperforge_mcp/http.py:478-484` writes certificate material with `NamedTemporaryFile(delete=False)`.
- No subsequent unlink operation was found.

**Impact**

Private client certificates persist in the temporary directory and accumulate. Processes with equivalent privileges, diagnostic bundles, backups, or container snapshots may recover them.

**Remediation**

Delete the file immediately after `SSLContext.load_cert_chain()` in a `finally` block, enforce restrictive file permissions, and avoid disk-backed key material where the TLS library permits it.

## Deployment-Dependent Risks

### Trusted Header Authentication

`hyperforge/src/hyperforge/api/authentication.py:33-94` accepts roles and users from `X-STF-*` and `X-NUCLIADB-*` headers without validating a signature or token. This is safe only if a trusted authorizer is the sole network peer and always strips client-supplied copies before injecting verified values. Direct API exposure would permit role spoofing. Enforce the trust boundary with network policy, mTLS, and proxy header sanitation; fail closed if the trusted authorizer is absent.

### Restricted Python Isolation

RestrictedPython is not an operating-system sandbox. Network verification is disabled by default in `agents/restricted/src/hyperforge_restricted/sandbox.py:25-42`, and debug execution can run in-process. No specific language escape was confirmed, but untrusted code should execute in a separate non-root container or microVM with no network, a read-only minimal filesystem, syscall restrictions, CPU/memory/process limits, and strict timeouts.

### Frontend OAuth Link Scheme

The frontend places an agent-provided OAuth URL directly into an anchor at `hyperforge/frontend/src/components/chat/ChatPanel.vue:72-77`, populated at `hyperforge/frontend/src/stores/chat.ts:184-190`. Vue escapes HTML, but URL schemes are not visibly allowlisted. Restrict links to `https:` and expected OAuth origins to prevent unsafe or deceptive schemes if a compromised integration supplies the URL.

### Secret Classification Gaps

Driver encryption covers declared `encrypted_fields`, but generic HTTP headers, external-agent headers, MCP certificates, and MCP stdio environment values are not consistently marked sensitive. Define a single secret-field policy, encrypt all persisted credentials, exclude them from API responses, and prevent them from entering logs or telemetry.

## Positive Controls

- SQL access reviewed uses SQLAlchemy expression construction and bound values; no concrete SQL injection was found.
- Runtime YAML loading uses `yaml.safe_load`; no pickle or equivalent unsafe deserialization was found.
- Vue templates reviewed use escaped interpolation rather than raw HTML rendering.
- Standalone JWT validation restricts algorithms and validates signature, expiration, optional not-before, issuer, audience, and scopes.
- The shared HTTP transport blocks IPv4 private and other non-global addresses, though HF-03 documents important gaps.
- Driver secret encryption uses Fernet and export encryption uses PBKDF2-derived keys.
- GitHub Actions references reviewed are pinned to full commit SHAs and checkout commonly disables credential persistence.
- Dedicated TruffleHog, Black Duck, Polaris, and zizmor workflows exist.
- `uv.lock` contains package hashes and `uv lock --check` passed.
- Common environment/credential files are ignored, and no tracked `.env`, private key, certificate, keystore, or Terraform state file was found.
- Renovate is configured in `.github/renovate.json`.

## Scope and Method

The review covered:

- 22 Python workspace manifests and the root `uv.lock` (185 resolved packages).
- The Vue/Vite frontend manifest and lockfile.
- FastAPI HTTP, WebSocket, MCP, OAuth, standalone UI, authentication, database, encryption, and session code.
- HTTP, external, MCP, NucliaDB, Google, Perplexity, and restricted-code agent security boundaries.
- Dockerfile and 25 GitHub Actions workflows.
- Current-tree secret patterns and tracked sensitive filenames.
- Targeted searches for injection, process execution, deserialization, SSRF, XSS, file operations, and unbounded I/O.

The worktree already contained uncommitted changes to:

- `hyperforge/src/hyperforge/standalone/app.py`
- `hyperforge/src/hyperforge/standalone/settings.py`

Those changes add configurable static-folder handling and were included in the reviewed state. They were not modified by this assessment.

## Verification Results

| Check | Result |
|---|---|
| `uv lock --check` | Passed; 185 packages resolved |
| `npm audit --json` | 6 advisories: 2 high, 1 moderate, 3 low |
| `npm audit --omit=dev --json` | 5 advisories: 2 high, 1 moderate, 2 low |
| `npm run build` | Passed with Vite 7.3.5 |
| `uv run pytest tests -q` in `agents/http` | 1 passed |
| `uv run pytest tests -q` in `agents/external` | 1 passed |
| `uv run pytest tests -q` in `agents/mcp` | 18 passed |
| `uv run pytest hyperforge/tests -q` | 159 passed; 48 setup errors because the Docker daemon was unavailable |
| Combined multi-package pytest invocation | Not valid due to duplicate `tests.conftest` module names; packages were rerun separately |
| Current-tree secret pattern scan | No confirmed production credential found |
| Tracked sensitive-file scan | No tracked matching files found |

## Limitations

- This was not a live penetration test and did not send exploit payloads to a deployed environment.
- Python advisory scanners (`pip-audit`, OSV Scanner), TruffleHog, Bandit, Semgrep, Syft, and Grype were not installed locally. Python vulnerabilities and full Git history secrets were therefore not independently queried.
- The Docker client was installed but the daemon was unavailable, so the image was not built or scanned and integration tests requiring PostgreSQL/Valkey containers could not start.
- Current-tree secret pattern matching cannot prove that all 304 repository commits are clean.
- Cloud, Kubernetes, reverse-proxy, firewall, mTLS, branch protection, GitHub environment, and secret-scope settings were not available for review.
- Dependency advisories describe affected versions; reachability judgments are based on static call-site review rather than runtime instrumentation.

## Remediation Roadmap

### Within 24 Hours

- Restrict standalone and internal API network exposure.
- Fix HF-01 and HF-02 or disable the affected routes.
- Rotate exposed credentials where network exposure cannot be ruled out.

### Within 7 Days

- Correct SSRF handling across all configurable outbound requests.
- Add request/response limits for MCP and HTTP agents.
- Add `.dockerignore`, a least-privilege runtime user, and a minimal runtime image.
- Upgrade vulnerable frontend dependencies and add npm SCA to CI.

### Within 30 Days

- Run Python SCA, full-history secret scanning, SAST, and a built-image vulnerability scan in CI with blocking severity policies.
- Harden restricted-code execution with OS-level isolation.
- Add negative security tests for anonymous configuration access, internal route access, role-header spoofing, IPv6/private redirects, DNS rebinding, oversized payloads, and secret redaction.
- Review logging and telemetry for prompts, model context, external responses, OAuth data, and credentials.

## Conclusion

The project has several sound foundations, including parameterized database access, safe YAML parsing, encrypted driver fields, pinned GitHub Actions, and dedicated security workflows. However, HF-01 and HF-02 create direct credential-disclosure paths, and HF-03 can turn configuration control into internal network access. These issues should be treated as release blockers for any deployment reachable by untrusted clients.
