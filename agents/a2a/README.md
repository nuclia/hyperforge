# hyperforge_a2a

A2A client agent for Hyperforge. It connects to an external Agent2Agent (A2A)
server and exposes streamed text responses as Hyperforge context.

`source` accepts either a gRPC address such as `localhost:8034` or an HTTP(S)
base URL such as `http://localhost:9999`. For HTTP(S) sources, the client
resolves `/.well-known/agent-card.json` and uses the transport advertised by
the remote Agent Card.
