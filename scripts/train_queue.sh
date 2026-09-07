#!/usr/bin/env bash
# Each .txt job contains one Hydra override argument per line.
set -uo pipefail
source "$(dirname "$0")/lib/queue.sh"
readonly QUEUE_DIR='queue/train'
readonly LOGS='outputs/queue/train'
readonly SUFFIX='.txt'
gpus=(0 1)
usage() { echo "usage: $0 [-g 0,1] [Hydra overrides ...]" >&2; exit 2; }
while getopts ':g:' opt; do
  case "$opt" in
    g) IFS=',' read -r -a gpus <<<"$OPTARG" ;;
    *) usage ;;
  esac
done
shift $((OPTIND - 1))
common=("$@")
mkdir -p "$QUEUE_DIR"
job_name() { basename "${1%"$SUFFIX".running}"; }
run_job() {
  local line
  local overrides=()
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    overrides+=("$line")
  done < "$1"
  uv run python -m pipelines.train "${common[@]+"${common[@]}"}" "${overrides[@]}"
}
queue_run "${gpus[@]}"
