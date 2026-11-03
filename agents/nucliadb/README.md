# NucliaDB Hyperforge agents

Basic ASK - Basic initial agent
Advanced ASK (deprecated)
ASK (experimental)
Sync (Sync Service oauth linked)

## Client identification

NucliaDB `/ask` and `/find` requests made by these agents use
`X-NDB-Client: agentic`, including when invoked as tools by a smart agent in a
RAO workflow. The Sync driver inherits the same behavior. Other request headers
are preserved, and caller-provided headers are not modified.

Workflow memory searches through `/find` also use the `agentic` client type.
The receiving NucliaDB API must support the `agentic` client type.