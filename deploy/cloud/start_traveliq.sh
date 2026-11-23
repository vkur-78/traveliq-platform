#!/usr/bin/env bash
set -euo pipefail

echo "[TRAVELIQ Cloud] Starting internal Flight Price API on 127.0.0.1:5001..."
gunicorn \
  --bind 127.0.0.1:5001 \
  --workers 1 \
  --threads 4 \
  --timeout 60 \
  --access-logfile - \
  --error-logfile - \
  app.app:app &
FLIGHT_PID=$!

echo "[TRAVELIQ Cloud] Starting internal Gender Classification API on 127.0.0.1:5002..."
gunicorn \
  --bind 127.0.0.1:5002 \
  --workers 1 \
  --threads 4 \
  --timeout 30 \
  --access-logfile - \
  --error-logfile - \
  app.gender_api:app &
GENDER_PID=$!

cleanup() {
  echo "[TRAVELIQ Cloud] Shutting down internal APIs..."
  kill "${FLIGHT_PID}" "${GENDER_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for_health() {
  local name="$1"
  local url="$2"
  local retries=45
  for i in $(seq 1 "${retries}"); do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      echo "[TRAVELIQ Cloud] ${name} is healthy (${url})."
      return 0
    fi
    sleep 1
  done
  echo "[TRAVELIQ Cloud] ERROR: ${name} failed health check after ${retries}s (${url})." >&2
  return 1
}

wait_for_health "Flight Price API" "http://127.0.0.1:5001/health"
wait_for_health "Gender Classification API" "http://127.0.0.1:5002/health"

PUBLIC_PORT="${PORT:-7860}"
echo "[TRAVELIQ Cloud] Launching TRAVELIQ Streamlit Web Platform on 0.0.0.0:${PUBLIC_PORT}..."
exec streamlit run streamlit_app.py \
  --server.address=0.0.0.0 \
  --server.port="${PUBLIC_PORT}" \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false \
  --browser.gatherUsageStats=false
