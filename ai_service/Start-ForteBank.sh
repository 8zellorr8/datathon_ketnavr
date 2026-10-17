#!/usr/bin/env sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"
if ! command -v python3 >/dev/null 2>&1; then
    printf '%s\n' 'Python 3.10+ (64 bit) is required. Python 3.12 is recommended.'
    printf '%s\n' 'Install Python from https://www.python.org/downloads/ or your operating system package manager.'
    exit 1
fi
exec python3 "$SCRIPT_DIR/Start-ForteBank.py" "$@"
