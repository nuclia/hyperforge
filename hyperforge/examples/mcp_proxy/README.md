# MCP proxy example (opencode)

Runs `hyperforge-mcp-proxy` in front of three upstream MCP servers and wires it
into opencode. The agent only sees two tools, `search` and `execute`.

Upstreams (`proxy.yaml`; paths are relative to the repository root):

- `inventory`: tiny FastMCP server in `inventory_server.py` (`get_inventory_stock`, `list_inventory_skus`)
- `everything`: `@modelcontextprotocol/server-everything`
- `filesystem`: `@modelcontextprotocol/server-filesystem`, limited to this directory

## Setup

```sh
# from the repo root
uv sync
npm --prefix hyperforge/examples/mcp_proxy/node install
```

The npm servers are installed locally instead of run via `npx -y`, so that one
lockfile works with registries that enforce a release cooldown. If your
registry rejects newer transitive dependencies, add `--before=<date>`.

## Run

Run the commands from this directory. `opencode.json` starts the proxy over stdio:

```sh
uv run --no-sync python smoke.py   # direct MCP client sanity check
opencode mcp list               # hyperforge-proxy  connected
opencode run "Use hyperforge-proxy search to find an inventory tool, then execute it for SKU WIDGET-42."
```

To use HTTP instead, start the proxy and then set `hyperforge-proxy-http.disabled` to `false`:

```sh
uv run --no-sync --directory ../../.. hyperforge-mcp-proxy \
  --config hyperforge/examples/mcp_proxy/proxy.yaml --transport http --host 127.0.0.1 --port 8765
```

`codemode: false` exposes `search` and `execute` as native tools
(`hyperforge-proxy_search`, `hyperforge-proxy_execute`). The proxy already
handles discovery, so wrapping it in opencode's Code Mode would add a second
discovery layer.
