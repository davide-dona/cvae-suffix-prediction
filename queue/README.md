# Batch queues

Jobs staged for a batch run, one folder per pipeline. Copy in what to run, then hand the whole
folder to the machine's GPUs.

## Training

```bash
cp config/datasets/bpic17.yaml config/datasets/bpic19.yaml queue/train/
scripts/train_queue.sh -m config/models/cvae.yaml
```

A job is a dataset config, and is named here by the config's own filename. One architecture per
sweep, so comparing both means running the script again with `-m config/models/transformer.yaml`.

## Generation

```bash
cp outputs/checkpoints/best/bpic17/cvae/*.pt queue/generate/
scripts/generate_queue.sh   # -n 100 for every job in the batch, -d <device> to override
```

A job is a copy of a best checkpoint, since a checkpoint carries the config and the run identity
of what wrote it and generation reads both from inside it. A best checkpoint is named after its
run's tag alone, so what it is called here says nothing about which dataset or model it holds;
the run inside it is what names its terminal lines and its log, whatever the copy is called.

## What happens to a job

One job runs per GPU at a time, and a GPU that finishes picks up the next rather than waiting on
the job beside it.

A job running right now is renamed `<name>.running`, and only a plain `.yaml` or `.pt` is ever
picked up, so nothing is run twice.

A job that succeeded is deleted from here. For a generation job that removes only the copy, never
the checkpoint under `outputs/checkpoints/best/`. One that failed is renamed `<name>.failed` and
stays, since a job a freed GPU could claim again would be retried on the spot rather than looked
at. Its output is under `outputs/queue/train/` or `outputs/queue/generate/`, named in the summary
the script prints. To try it again, drop the suffix:

```bash
mv queue/train/bpic17.yaml.failed queue/train/bpic17.yaml
```

A `.running` left behind after the script is gone is an interrupted job, and is re-queued the same
way.

Everything here is gitignored except this file.
