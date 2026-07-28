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

A run writes two things:

- `outputs/tensorboard/`: the loss and its terms under `train/` and `val/`, plus `kl_weight`.
  ```bash
  tensorboard --logdir outputs/tensorboard
  ```
- `outputs/checkpoints/best-models/best-model-epoch-<n>.pt`: written on every validation improvement. The hyperparameters travel with the weights, so a checkpoint can be reloaded without restating them.

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
  device: mps
```
