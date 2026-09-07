#!/usr/bin/env bash
# Pull a training VM's generations back onto this machine.
#
# Training curves are logged to W&B, and a finished run uploads its checkpoint there as an
# Artifact, so the only VM-side output left to fetch is what `pipelines.generate` writes;
# evaluate and visualize already run locally against it.
#
# This is a plain one-way rsync, pulling whatever is new or changed and never deleting
# anything on this side.
set -euo pipefail

usage() {
  echo "usage: $0 [-n] <ssh-host> [<remote-repo-path>]" >&2
  echo "  -n   dry run: show what would be pulled without copying anything" >&2
  exit 2
}

dry_run=()
while getopts ':n' opt; do
  case "$opt" in
    n) dry_run=(--dry-run) ;;
    *) usage ;;
  esac
done
shift $((OPTIND - 1))

[[ $# -ge 1 ]] || usage
host="$1"
remote_path="${2:-suffix-generation}"

rsync -avz --progress "${dry_run[@]+"${dry_run[@]}"}" \
  "$host:$remote_path/outputs/generations/" outputs/generations/
