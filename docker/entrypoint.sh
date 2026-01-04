#!/usr/bin/env bash
set -euo pipefail

cd /workspace/MemFlow

HOST="${MEMFLOW_HOST:-0.0.0.0}"
PORT="${MEMFLOW_PORT:-7860}"
EXTRA_ARGS="${MEMFLOW_ARGS:-}"

exec python webui.py --host "${HOST}" --port "${PORT}" ${EXTRA_ARGS}

