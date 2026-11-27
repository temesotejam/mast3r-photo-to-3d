#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "$0")/.."
venv_dir="${HOME}/.local/share/mast3r-photo-to-3d/venv"
python_bin="$venv_dir/bin/python"

if [[ "${1:-}" == "install" ]]; then
    python3 -m venv "$venv_dir"
    exec "$python_bin" scripts/setup_cpu.py
fi

if [[ ! -x "$python_bin" ]]; then
    echo "Run bash scripts/start-codespace.sh install to prepare the environment." >&2
    exit 1
fi

health_url=http://127.0.0.1:7861/api/health
if curl --fail --silent --max-time 2 "$health_url" >/dev/null; then
    echo "MASt3R is already running on http://127.0.0.1:7861"
    exit 0
fi

log_file="${TMPDIR:-/tmp}/mast3r-photo-to-3d.log"
nohup "$python_bin" -u server.py >"$log_file" 2>&1 </dev/null &
server_pid=$!
for _ in {1..120}; do
    if curl --fail --silent --max-time 2 "$health_url" >/dev/null; then
        echo "Open port 7861 in your browser: http://127.0.0.1:7861"
        exit 0
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        cat "$log_file" >&2
        exit 1
    fi
    sleep 1
done
echo "MASt3R is still starting. Check $log_file and the Ports tab for port 7861." >&2
exit 1
