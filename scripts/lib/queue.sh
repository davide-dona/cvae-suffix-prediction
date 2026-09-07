# The queue a batch script runs on: one job per GPU at a time, claimed by an atomic rename.
#
# A job is claimed by renaming it `<suffix>.running`, deleted once it exits 0, and renamed
# `<suffix>.failed` when it does not. Only a plain `<suffix>` is ever claimed, so a failure stays
# in the folder to be looked at rather than being picked straight back up by the next free GPU.
#
# Sourced, not run. Bash has no way to declare an interface, so the contract is stated here: a
# caller sets QUEUE_DIR, LOGS and SUFFIX, defines job_name and run_job, and calls
# `queue_run "${gpus[@]}"`.
#
#   QUEUE_DIR          the folder jobs are staged in
#   LOGS               the folder a job's output is written to
#   SUFFIX             what a staged job's filename ends in, `.yaml` or `.pt`
#   job_name <claimed> what the terminal lines and the log filename call this job
#   run_job <claimed>  the command to run, with its output already redirected to the log

# Take the next job off the queue. `mv` is a rename, so exactly one worker wins a given file.
claim() {
  local candidate
  [[ -e "$stop" ]] && return 1
  for candidate in "$QUEUE_DIR"/*"$SUFFIX"; do
    if mv "$candidate" "$candidate.running" 2>/dev/null; then
      printf '%s' "$candidate.running"
      return 0
    fi
  done
  return 1
}

# Run every job this GPU manages to claim, until the queue is empty. `CUDA_VISIBLE_DEVICES` masks
# in one GPU, so the profile's `cuda:0` is this worker's GPU. It is exported rather than prefixed
# onto the command because `run_job` is a shell function, and a prefix assignment on one of those
# is scoped differently in POSIX mode than out of it.
worker() {
  local gpu="$1" claimed name log started elapsed status
  export CUDA_VISIBLE_DEVICES="$gpu"
  while claimed="$(claim)"; do
    name="$(job_name "$claimed")"
    log="$LOGS/$name-$(date +%Y%m%d-%H%M%S).log"
    printf '[gpu %s] start  %-32s -> %s\n' "$gpu" "$name" "$log"
    started="$SECONDS"
    run_job "$claimed" >"$log" 2>&1
    status=$?
    elapsed=$(( SECONDS - started ))
    if (( status == 0 )); then
      rm -f "$claimed"
      printf '[gpu %s] ok     %-32s (%dh %02dm)\n' \
        "$gpu" "$name" $(( elapsed / 3600 )) $(( elapsed % 3600 / 60 ))
      echo "ok $name" >>"$results"
    else
      mv "$claimed" "${claimed%.running}.failed"
      printf '[gpu %s] FAILED %-32s (%dh %02dm, exit %d) see %s\n' \
        "$gpu" "$name" $(( elapsed / 3600 )) $(( elapsed % 3600 / 60 )) "$status" "$log"
      echo "failed $name $log" >>"$results"
    fi
  done
  return 0
}

# Work the queue with one worker per GPU, then print what became of every job.
#
# Args:
#   the GPU indices to run on, one worker each.
# Returns:
#   0 if every job exited 0, 1 if the queue was empty or any job failed.
queue_run() {
  local gpu ok_count failed_count

  shopt -s nullglob
  local queued=("$QUEUE_DIR"/*"$SUFFIX")
  (( ${#queued[@]} )) || { echo "nothing queued in $QUEUE_DIR/" >&2; return 1; }

  mkdir -p "$LOGS"
  results="$(mktemp)"

  # Interrupting stops the queue rather than the script. Ctrl-C reaches the jobs in flight through
  # the process group and ends them; this is what stops a freed GPU from starting the next job,
  # so the summary is still printed and what never ran is still queued.
  stop="$results.stop"
  trap 'touch "$stop"' INT TERM

  for gpu in "$@"; do
    worker "$gpu" &
  done
  # A caught signal interrupts `wait`, so keep waiting until the workers are actually done.
  until wait; do :; done

  ok_count=$(grep -c '^ok ' "$results" || true)
  failed_count=$(grep -c '^failed ' "$results" || true)
  printf '\nSummary: %d ok, %d failed\n' "$ok_count" "$failed_count"
  grep '^failed ' "$results" | awk '{ printf "  %-32s %s\n", $2, $3 }'
  (( failed_count )) && echo "Re-queue a failure by dropping the .failed suffix in $QUEUE_DIR/."
  rm -f "$results" "$stop"
  (( failed_count == 0 ))
}
