#!/usr/bin/env bash
# Evaluate every generations file that has no report yet, one at a time. Evaluation is CPU-bound
# and already parallelizes internally via pipelines.evaluate's own -j/--workers, so this is a plain
# sequential loop rather than scripts/lib/queue.sh's per-GPU worker queue.
#
# Both generations trees are walked: outputs/generations/ and the same layout under pinned/, where
# a run is moved by hand so a wipe of outputs/ cannot touch it. A report is named after the run
# that wrote it either way, so both land under outputs/eval/ and a pinned run is no different to
# anything reading one.
set -uo pipefail

readonly GENERATIONS_DIRS=('outputs/generations' 'pinned/generations')
readonly EVAL_DIR='outputs/eval'
readonly LOGS='outputs/queue/evaluate'
force=0
workers=()

usage() {
  echo "usage: $0 [-f] [-j <workers>]" >&2
  exit 2
}

while getopts ':fj:' opt; do
  case "$opt" in
    f) force=1 ;;
    j) workers=(-j "$OPTARG") ;;
    *) usage ;;
  esac
done

mkdir -p "$LOGS"

# pinned/ is gitignored, so a checkout that has never pinned a run does not have it.
roots=()
for dir in "${GENERATIONS_DIRS[@]}"; do
  [[ -d "$dir" ]] && roots+=("$dir")
done
(( ${#roots[@]} )) || { echo "none of ${GENERATIONS_DIRS[*]}/ exists" >&2; exit 1; }

generations=()
while IFS= read -r -d '' file; do
  generations+=("$file")
done < <(find "${roots[@]}" -type f -name '*.parquet' -print0 | sort -z)
(( ${#generations[@]} )) || { echo "nothing under ${roots[*]}/" >&2; exit 1; }

ok=0
failed=()

for file in "${generations[@]}"; do
  # Whichever tree it came from, a run names itself the same way below the root.
  rel="$file"
  for dir in "${GENERATIONS_DIRS[@]}"; do
    rel="${rel#"$dir"/}"
  done
  name="${rel%.parquet}"
  report="$EVAL_DIR/$name.json"

  if (( ! force )) && [[ -e "$report" ]]; then
    echo "skip   $name (already evaluated, see $report)"
    continue
  fi

  log="$LOGS/${name//\//_}-$(date +%Y%m%d-%H%M%S).log"
  echo "start  $name -> $log"
  if uv run python -m pipelines.evaluate -g "$file" "${workers[@]+"${workers[@]}"}" 2>&1 | tee "$log"; then
    echo "ok     $name"
    ok=$((ok + 1))
  else
    echo "FAILED $name (see $log)"
    failed+=("$name")
  fi
done

printf '\nSummary: %d ok, %d failed\n' "$ok" "${#failed[@]}"
for name in "${failed[@]+"${failed[@]}"}"; do
  echo "  $name"
done
(( ${#failed[@]} == 0 ))
