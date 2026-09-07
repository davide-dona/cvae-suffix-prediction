#!/usr/bin/env bash
# Each .pt job is a copy of a source checkpoint.
set -uo pipefail
source "$(dirname "$0")/lib/queue.sh"
readonly QUEUE_DIR='queue/generate'
readonly LOGS='outputs/queue/generate'
readonly SUFFIX='.pt'
gpus=(0 1)
usage() { echo "usage: $0 [-g 0,1] [Hydra overrides ...]" >&2; exit 2; }
while getopts ':g:' opt; do
  case "$opt" in
    g) IFS=',' read -r -a gpus <<<"$OPTARG" ;;
    *) usage ;;
  esac
done
shift $((OPTIND - 1))
overrides=("$@")
mkdir -p "$QUEUE_DIR"
job_name() { basename "${1%"$SUFFIX".running}"; }
run_job() {
  uv run python -m pipelines.generate "checkpoint=\"$1\"" "${overrides[@]+"${overrides[@]}"}"
}
queue_run "${gpus[@]}"
