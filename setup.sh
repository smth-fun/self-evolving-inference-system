#!/bin/bash
set -e

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "Creating venv..."
    uv venv --python=3.12
    source .venv/bin/activate
    echo "Installing dependencies (this may take a few minutes)..."
    uv pip install -e ./mini-sglang
    echo "Done. venv ready at .venv"
else
    echo ".venv already exists, skipping."
fi
