#!/usr/bin/env bash
# Score completed generation artifacts, skipping hashes with completed reports.
set -uo pipefail
uv run python -m scripts.evaluate_queue "$@"
