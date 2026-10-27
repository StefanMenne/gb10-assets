---
name: gb10-setup
description: Check and fix the connection to the gb10-assets server, see what runs on it and get the log of a failed job. Use when gb10-assets tools fail, are missing, report "not reachable", 401, 403 or a version mismatch, or when the user asks about the server state or the plugin setup.
---

# gb10-assets setup and troubleshooting

The plugin starts the MCP server with `uvx` from the GitHub repo and passes the plugin options as
`GB10_ASSETS_URL` and `GB10_ASSETS_TOKEN`. Change them with `/plugin` (gb10-assets, configure) or
`claude plugin configure gb10-assets`; restart Claude Code afterwards.

| Symptom | Cause and fix |
|---|---|
| no gb10-assets tools | `uv` missing (`curl -LsSf https://astral.sh/uv/install.sh \| sh`) or the server URL unset; see `/mcp` |
| `not reachable` | wrong URL or port, container down: on the GB10 `scripts/docker.sh up`; `curl <url>/health` |
| `401` | token missing or wrong: `GB10_ASSETS_TOKEN` in `~/gb10-assets/.env` on the GB10 system |
| `403` on `blender` with `code` | the server disables own code (`BLENDER_CODE=0` in its `.env`); use a bundled script |
| version mismatch | client and server differ: `uv cache clean gb10-assets` and restart, or update the server (`git pull`, `scripts/docker.sh build`, `up`) |
| job `failed` | the tool is not installed (`scripts/docker.sh install <tool>`) or ran out of memory; read the log |

Server state and logs (if the `gb10-assets` CLI is installed, otherwise the same with `uvx --from
"gb10-assets[client] @ git+https://github.com/StefanMenne/gb10-assets" gb10-assets ...`):

```sh
gb10-assets health          # version, running job, queue
gb10-assets log <job-id>    # log of a failed job
```

The server runs one GPU job at a time; `job_status` shows the queue position.
