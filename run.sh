#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

COMPOSE_CMD="${COMPOSE_CMD:-docker compose}"
QDRANT_SERVICE="${QDRANT_SERVICE:-qdrant}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Threadfall launcher

Starts qdrant via docker compose if needed, waits until it responds, then runs Streamlit.

Environment variables:
  COMPOSE_CMD      default: "docker compose"
  QDRANT_SERVICE   default: "qdrant"
  QDRANT_URL       default: "http://localhost:6333"
  STREAMLIT_PORT   default: "8501"
  PYTHON           default: ".venv/bin/python" if present, else "python3"
EOF
  exit 0
fi

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON="$ROOT_DIR/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    return 1
  fi
}

wait_for_http_200() {
  local url="$1"
  local timeout_seconds="${2:-45}"
  local start
  start="$(date +%s)"
  while true; do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    if (( "$(date +%s)" - start >= timeout_seconds )); then
      echo "Timed out waiting for $url" >&2
      return 1
    fi
    sleep 1
  done
}

echo "Threadfall launcher"

require_cmd docker
require_cmd curl

if ! docker info >/dev/null 2>&1; then
  echo "Docker does not appear to be running. Start Docker Desktop and re-run." >&2
  exit 1
fi

echo "Checking Qdrant service..."
running_services="$($COMPOSE_CMD ps --status running --services 2>/dev/null || true)"
qdrant_running="false"
while IFS= read -r svc; do
  if [[ "$svc" == "$QDRANT_SERVICE" ]]; then
    qdrant_running="true"
    break
  fi
done <<<"$running_services"

if [[ "$qdrant_running" == "true" ]]; then
  echo "Qdrant is already running."
else
  echo "Starting Qdrant via docker compose..."
  $COMPOSE_CMD up -d "$QDRANT_SERVICE"
fi

echo "Waiting for Qdrant to become ready..."
if ! wait_for_http_200 "${QDRANT_URL%/}/readyz" 35; then
  # Older builds may not expose /readyz; fall back to a lightweight endpoint.
  wait_for_http_200 "${QDRANT_URL%/}/collections" 35
fi
echo "Qdrant is ready at $QDRANT_URL"

echo "Launching Streamlit on port $STREAMLIT_PORT..."
exec "$PYTHON" -m streamlit run app.py --server.port "$STREAMLIT_PORT"

