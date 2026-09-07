import argparse

from huggingface_hub import snapshot_download

from scripts.hub import HF_REPO_ID
from src import paths


def run() -> None:
    """Download every published model into `pretrained/`, mirroring the Hugging Face repo's
    `dataset/model.pt` layout exactly.

    These are the checkpoints `scripts/publish.py` proposed and a maintainer merged. A run's own
    checkpoints live under `outputs/checkpoints/` instead.
    """
    snapshot_download(repo_id=HF_REPO_ID, repo_type='model', local_dir=paths.PRETRAINED_DIR)


def main() -> None:
    argparse.ArgumentParser(
        description='Fetch every published model from the Hugging Face repo into `pretrained/`.'
    ).parse_args()

    run()
    print(f'Fetched every published model into {paths.PRETRAINED_DIR}')


if __name__ == '__main__':
    main()
