# Sandbox Policy

This policy defines safe execution rules for optional code execution.

## Default mode

- `allow_execution = false` (static checks only)
- network access disallowed by policy
- execution should be limited by timeout/memory if enabled
- non-root and restricted filesystem required in production container deployments

## Static blocklist

Blocked imports include:

- `os`
- `subprocess`
- `socket`
- `requests`
- `http`
- `pathlib`
- `shutil`

## If execution is enabled

Use constrained runtime with:

- no network
- strict CPU timeout
- memory limit
- non-root user
- restricted filesystem
- telemetry for timeout/exception/resource usage

## Telemetry

Execution telemetry is written to:

- `logs/sandbox_telemetry.jsonl`
