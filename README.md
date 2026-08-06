# C-VAE for Suffix Generation
In this repository, we present a Conditional Variational Autoencoder (C-VAE) model designed for generating suffixes based on given prefixes.
We provide a comprehensive implementation of the C-VAE architecture, along with training scripts, evaluation metrics, configuration files, and pre-trained models to facilitate research and experimentation in the field of predictive process monitoring.


## Install
Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync                    # installs the locked dependencies into .venv
```

## Reproducibility
The repository is designed to ensure reproducibility of results. To reproduce the experiments, follow these steps:


## Run
A dataset is a raw log at `data/<name>/original.csv` plus a `config/<name>.yaml`. Every pipeline below takes that config with `-c` and reads everything else from it.

### Preprocessing
Run once per dataset, before anything else. It writes the splits and dataset description under `data/<name>/processed/` and the discovered declarative model to `data/<name>/declare/model.decl`:

```bash
python -m pipelines.preprocess -c config/sepsis.yaml
```

### Training
```bash
python -m pipelines.train -c config/sepsis.yaml
```

Training and generation read those outputs and stop with an error naming what is missing if the dataset has not been preprocessed.

A run writes two things, both named by the same `<dataset>/<experiment_name>-<timestamp>` run name, so a run's curves and its result are found under one name:

- `outputs/tensorboard/<dataset>/<experiment_name>-<timestamp>/`: the loss and its terms under `train/` and `val/`, plus `kl_weight`. 
Point TensorBoard at the root and every run shows up as its own toggleable set of curves, grouped by dataset:
  ```bash
  tensorboard --logdir outputs/tensorboard
  ```
- `outputs/best-models/<dataset>/<experiment_name>-<timestamp>.pt`: the run's result. One file, overwritten every time the validation loss improves, so the last improvement of the run is what is left in it.

### Inference

### Evaluation

### Configs