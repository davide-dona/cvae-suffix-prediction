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

## Run
A dataset is a raw log at `data/<name>/original.csv` plus a `config/<name>.yaml`. Both pipelines below take that config with `-c` and read everything else from it.

### Training
```bash
python -m pipelines.train -c config/sepsis.yaml
```

The raw log is preprocessed first if `data/<name>/processed/` holds no splits. To run that step alone:

```bash
python -m pipelines.preprocess -c config/sepsis.yaml
```

Training stops at `training.max_num_epochs`, or once the validation loss plateaus for `early_stopping.patience` evaluations. Only epochs at the full KL weight count for early stopping and best-model selection, since a partly annealed weight lowers the loss for free.

A run writes three things, all three named by the same `<dataset>/<experiment_name>-<timestamp>` run name, so a run's curves, its history and its result are found under one name:

- `outputs/tensorboard/<dataset>/<experiment_name>-<timestamp>/`: the loss and its terms under `train/` and `val/`, plus `kl_weight`. Point TensorBoard at the root and every run shows up as its own toggleable set of curves, grouped by dataset:
  ```bash
  tensorboard --logdir outputs/tensorboard
  ```
  The Scalars dashboard keeps one chart per tag, which is what makes a metric comparable across runs; the Custom Scalars dashboard has the same numbers again as one chart per metric, with that run's train and val curve drawn together.
  The run directory is printed when training starts. Two runs of the same dataset are told apart by `experiment_name`, so override it in a dataset config when you want a run labelled by what you changed.
- `outputs/best-models/<dataset>/<experiment_name>-<timestamp>.pt`: the run's result. One file, overwritten every time the validation loss improves, so the last improvement of the run is what is left in it. The hyperparameters travel with the weights, along with the epoch and validation loss they came from, so a checkpoint can be reloaded without restating them.
- `outputs/checkpoints/<dataset>/<experiment_name>-<timestamp>/epoch-<n>.pt`: the history behind that result, one file per validation improvement. Nothing is pruned, so a long run leaves a few dozen files of a few MB each.

### Inference

### Evaluation

### Configuration
`config/base.yaml` holds every hyperparameter shared across datasets. Each dataset file sits next to it and holds only its `data:` section. A run merges the two, with the dataset file taking precedence.

| Section | Holds |
| --- | --- |
| `data` | dataset directory, raw column names, split fractions, `max_seq_len`, batch size, dataloader workers |
| `model` | `embeddings`, `prefix_encoder`, `suffix_encoder`, `prior`, `latent`, `attention`, `decoder` |
| `loss` | the cyclical KL annealing schedule |
| `optimizer` | Adam learning rate and weight decay |
| `training` | epoch budget, gradient clipping, device, output directories, validation interval |
| `early_stopping` | patience and the minimum relative improvement that resets it |

A dataset file can override any base field, nested ones included, by repeating the key at the same level:

```yaml
# config/sepsis.yaml
training:
  device: cuda
```
