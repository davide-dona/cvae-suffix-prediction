#!/usr/bin/env bash
# Generate suffixes for every checkpoint sitting in queue/generate/, one run per GPU at a time.
#
# The queue itself, and what happens to a checkpoint that succeeds or fails, is
# scripts/lib/queue.sh. A staged checkpoint is a copy: deleting it on success leaves the one under
# outputs/checkpoints/best/ untouched.
set -uo pipefail

source "$(dirname "$0")/lib/queue.sh"

readonly QUEUE_DIR='queue/generate'
readonly LOGS='outputs/queue/generate'
readonly SUFFIX='.pt'
gpus=(0 1)
device=()
samples=()

usage() {
  echo "usage: $0 [-d <device>] [-g 0,1] [-n <samples>]" >&2
  exit 2
}

while getopts ':d:g:n:' opt; do
  case "$opt" in
    d) device=(-d "$OPTARG") ;;
    g) IFS=',' read -r -a gpus <<<"$OPTARG" ;;
    n) samples=(-n "$OPTARG") ;;
    *) usage ;;
  esac
done

mkdir -p "$QUEUE_DIR"

# A best checkpoint is named after the run's tag alone, so a copy of one says nothing about which
# dataset or model it is. The run it carries is what names it here, whatever it was copied as.
job_name() {
  uv run python -m scripts.run_name -m "$1" 2>/dev/null || basename "${1%"$SUFFIX".running}"
}

run_job() {
  uv run python -m pipelines.generate -m "$1" "${device[@]+"${device[@]}"}" "${samples[@]+"${samples[@]}"}"
}

queue_run "${gpus[@]}"
