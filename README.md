# C-VAE for Suffix Generation

A Conditional Variational Autoencoder (C-VAE) for generating suffixes based on given prefixes.

This repository provides a comprehensive implementation of the C-VAE architecture, along with training scripts, evaluation metrics, configuration files, and pre-trained models.

The whole project is built to support research and experimentation in **predictive process monitoring**.

---

## Install

**Requirements:** Python 3.13+, [uv](https://docs.astral.sh/uv/) and [Git LFS](https://git-lfs.com)

The datasets under `data/` are tracked with Git LFS. A plain clone
would only checks out pointer files. Install Git LFS once per machine and pull them before preprocessing:

```bash
git lfs install
git lfs pull
```

```bash
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv sync                     # installs the locked dependencies into .venv
```

Training logs metrics and checkpoints to [W&B](https://wandb.ai). Sign in once per machine:

```bash
uv run wandb login
```

---

## Reproducibility

The four pipelines below run in sequence, each reading what the previous one wrote. 

Preprocessing and training take `-c`/`--config`, the dataset's experiment config YAML (e.g. `config/datasets/bpic17.yaml`). 

Training takes `-m`/`--model`, the architecture to build (e.g. `config/models/cvae.yaml`), which also carries every non-dataset setting a run needs: the optimizer, the training loop, and, for the CVAE, the loss. Generation spells `-m` differently: there it is the trained checkpoint, a run's weights rather than an architecture, and takes an optional `-d`/`--device` to generate on a different machine than the one the run trained on.

### 1. Preprocessing

Run once per dataset, before anything else:

```bash
python -m pipelines.preprocess -c config/datasets/<dataset>.yaml
```

The out-of-time splits as well as the fitted codec and the declarative model are written to `data/<dataset>/`.

> [!WARNING]
> Training and generation read these outputs and will stop with an error naming what's missing if the dataset hasn't been preprocessed yet.

Every artifact is computed on each run: the continuation index of both held-out splits, read by training (to select checkpoints) and by evaluation, and the declarative model, read by evaluation to score conformance. Discovering the declarative model is the slowest step by a wide margin.

### 2. Training

Once the dataset is preprocessed, start a run:

```bash
python -m pipelines.train -m config/models/<architecture>.yaml -c config/datasets/<dataset>.yaml
```

`-m`/`--model` and `-c`/`--config` are both required. Two architectures are shipped: `config/models/cvae.yaml`, the conditional VAE, and `config/models/transformer.yaml`, the same backbone with the latent taken out. Which class gets built is read off `model.kind` inside the file. A run cannot be carried on from where it left off: one that finishes, is interrupted or dies is over, and the way to get more training is a fresh run.

#### Running a batch on every GPU at once

Testing an experiment usually means training it on every dataset, then generating from every run
it produced. Both are batched the same way: copy the jobs into `queue/` and hand the whole folder
to the machine's GPUs.

```bash
cp config/datasets/bpic17.yaml config/datasets/bpic19.yaml queue/train/
scripts/train_queue.sh -m config/models/cvae.yaml   # -g 0,1 by default

cp outputs/checkpoints/best/bpic17/cvae/*.pt queue/generate/
scripts/generate_queue.sh   # -n 100 for every job in the batch
```

A training job is a dataset config, run under the one architecture the script was given; a generation job is a copy of a best checkpoint, which carries
the config and the run identity of what wrote it. Both scripts are thin callers of
`scripts/lib/queue.sh`, which is the queue itself. Comparing both architectures over every dataset
is running `train_queue.sh` once per model, `-m config/models/cvae.yaml` and then
`-m config/models/transformer.yaml`, against the same queued datasets.

One job per GPU at a time, and a GPU that finishes picks up the next rather than waiting on the job
beside it. Each is launched with `CUDA_VISIBLE_DEVICES` masking in its own card, so the one profile
is used for both rather than a second one naming `cuda:1`.

Each job's output goes to `outputs/queue/<pipeline>/<job>-<timestamp>.log`, since two of them share
a terminal; the console gets a line per job and a summary at the end. A job that succeeded is
deleted from the queue, and one that failed is renamed `<job>.failed` and kept, to be re-queued by
dropping the suffix once the log has been read. See [`queue/README.md`](queue/README.md).

> [!NOTE]
> **Skip training:** pre-trained models are available on the Hugging Face model hub. Fetch every published model into `pretrained/` with:
> ```bash
> python -m scripts.fetch
> ```
> There is one file per model per log, at `pretrained/<name>/<model>.pt`.

Training curves are logged live to the `suffix-generation` W&B project; the run prints its URL as
soon as logging starts, so watching a VM's training needs no tunnel or synced files.

Runs are grouped in W&B by the experiment they belong to, `<name>/<model>`, and tagged with each half of it, so one model on one log reads as one group of runs and either axis can be filtered on alone.

A run's checkpoint is written to `outputs/checkpoints/best/<name>/<model>/<timestamp>.pt`, overwritten each time the selection score improves, so the file always holds the run's best step rather than its last. When the run finishes it is uploaded once, as a new version of the `<name>-<model>` Artifact aliased with the run's own timestamp:

```bash
wandb artifact get <name>-<model>:<timestamp>   # a particular run
wandb artifact get <name>-<model>:latest        # the most recent run of that experiment
```

One run leaves one version, so an experiment's Artifact lineage reads as its run history. A run that dies before finishing uploads nothing, and its checkpoint stays on the machine that trained it.

### 3. Inference

After training, generate suffixes for the test set:

```bash
python -m pipelines.generate -m <path-to-checkpoint>
```

- `-m`/`--checkpoint` points to the checkpoint to generate with, from `pretrained/` or `outputs/checkpoints/best/`.
- `-d`/`--device` overrides the device to generate on, e.g. to run on a different machine than the one the run trained on. Defaults to the run's own `training.device`.
- `-n`/`--num-samples` overrides how many suffixes are drawn per prefix for this generation alone. Defaults to the run's own `inference.evaluation_samples`.

The generated suffixes for every prefix of the test split are written to `outputs/generations/<name>/<model>/<timestamp>.parquet`, named after the run the checkpoint carries.

### 4. Evaluation

Reads the generated suffixes and writes an evaluation report:

```bash
python -m pipelines.evaluate -g <path-to-generations> -j <number-of-jobs>
```

- `-g`/`--generations` points to the generations file to score, produced by `pipelines.generate`.
- `-j`/`--workers` sets how many processes to score with, defaulting to one per available CPU. 

The resulting report is written to `outputs/eval/<name>/<model>/<timestamp>.json`, and the scores of each prefix behind it to the same path with a `.parquet` suffix.

### 5. Publishing

Once a run has been evaluated and is worth being the one others reach for, propose its best checkpoint as that model's published version:

```bash
python -m scripts.publish -m <path-to-best-checkpoint>
```

`-m`/`--checkpoint` points to the checkpoint to publish, from `outputs/checkpoints/best/`. Which run deserves the name is exactly the decision this step exists to record, so it is named rather than searched for.

The checkpoint file is uploaded as it sits, to `<name>/<model>.pt` in the Hugging Face model repo: it holds only what rebuilding the model reads, so there is nothing to trim off one first.

---

## Notebooks

---

## Visualization

Once a dataset has been evaluated, the scores of one or more runs can be plotted and tabulated with:

```bash
python -m pipelines.visualize -e <path-to-report> [<path-to-report> ...]
python -m pipelines.visualize -E outputs/eval
```

- `-e`/`--evaluations` takes the paths to the evaluation reports to compare, from `pipelines.evaluate`; passing several overlays them on the same axes, which is also how models or datasets are compared.
- `-E`/`--evaluations-dir` instead compares every report under a directory, at any depth: `outputs/eval` for a whole set of results, `outputs/eval/bpic17` for one dataset.

The figures are written to `outputs/visual/figures/` as PDF, which is what the paper takes, and the comparison tables to `outputs/visual/tables/` as `tex`. Each figure covers every dataset and model at once, so it is named after what it holds rather than after a run.

---

## Configs

A dataset config declares everything a run needs: where to find the raw log and how to read it, the model architecture, and every training and inference hyperparameter. All configs are validated against the models in `src/configs/schema/`, one module per section.

### Config inheritance

Fields can be overridden between files. Two layers are deep-merged in order, each taking precedence over the last:

1. `config/models/<architecture>.yaml` — the `-m`/`--model` architecture, e.g. `config/models/cvae.yaml`. Owns the whole `model` section, including the `kind` that says which class to build, and every other setting that does not vary with the dataset: the seed, the `dataloader`, the `optimizer`, the `training` loop, `early_stopping`, `inference`, and, for the CVAE, `model.loss`. An architecture is chosen independently of the log it runs on, which is why it is a layer rather than a block inside a dataset config; a different machine (a different device, batch size, learning rate) means a new model config variant rather than a third layer.
2. `config/datasets/<dataset>.yaml` — the `-c`/`--config` dataset config, e.g. `config/datasets/sepsis.yaml`. Owns the raw log and `declare`, the declarative-model discovery settings.
