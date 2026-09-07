import argparse
from pathlib import Path

from src import paths
from src.identity import RunIdentity, wandb_id
from src.model import load_checkpoint


def run(checkpoint_path: Path) -> None:
    """Print the identity of the run a checkpoint was written by, as one filename-safe name.

    Args:
        checkpoint_path: The checkpoint to read. Its run is read from inside it, so a copy of one
            is named after the run that wrote it whatever the copy is called.
    """
    print(wandb_id(RunIdentity.from_dict(load_checkpoint(checkpoint_path)['run'])))


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Print the identity of the run a checkpoint was written by, as '
        '<dataset>-<model>-<tag>.'
    )
    parser.add_argument(
        '-m',
        '--checkpoint',
        type=paths.existing_file,
        metavar='CHECKPOINT',
        required=True,
        help='Path to the checkpoint to read.',
    )
    args = parser.parse_args()

    run(args.checkpoint)


if __name__ == '__main__':
    main()
