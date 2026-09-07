import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, create_commit, file_exists, whoami
from huggingface_hub.errors import HfHubHTTPError, LocalTokenNotFoundError

from scripts.hub import HF_REPO_ID
from src import paths
from src.identity import RunIdentity
from src.model import CHECKPOINT_KEYS, load_checkpoint, require_keys


def _mebibytes(path: Path) -> str:
    """One file's size, in the unit a checkpoint is worth reading in."""
    return f'{path.stat().st_size / 1024**2:.1f} MiB'


def _account() -> str:
    """Check the Hugging Face credentials and return the account name a pull
    request would be opened under.

    Returns:
        The account name a pull request would be opened under.
    Raises:
        SystemExit: If there are no usable credentials, saying how to supply them.
    """
    try:
        return whoami()['name']
    except (LocalTokenNotFoundError, HfHubHTTPError) as error:
        raise SystemExit(
            f'not signed in to Hugging Face ({error}). Run "hf auth login", or set HF_TOKEN.'
        ) from error


def run(model_paths: list[Path]) -> None:
    """Propose one or more runs' checkpoints as published models on the Hugging Face Hub, as a
    single pull request, under a name carrying no run tag.

    The file is uploaded as it sits: a checkpoint holds only what rebuilding the model reads, so
    there is nothing to trim off one before it is published.

    Args:
        model_paths: The checkpoints to publish, from `outputs/checkpoints/best/`. Named rather
            than searched for: every run of one config is a candidate and choosing between them
            is the whole point of this step.
    Raises:
        SystemExit: If there are no Hugging Face credentials, or if the confirmation is declined.
        FileNotFoundError: If there is no checkpoint at one of `model_paths`.
        ValueError: If a checkpoint is missing a key rebuilding the model reads.
    """
    missing = [path for path in model_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f'no checkpoint at {", ".join(str(path) for path in missing)}.')

    account = _account()

    operations = []
    descriptions = []
    for model_path in model_paths:
        checkpoint = load_checkpoint(model_path)
        require_keys(
            checkpoint,
            CHECKPOINT_KEYS,
            subject=str(model_path),
            purpose='published',
            remedy='It was written by an older version of `save_checkpoint`; retrain, or publish '
            'a newer run.',
        )
        # The destination comes from the run's identity, not the checkpoint's filename, making it
        # invariant to local naming.
        run = RunIdentity.from_dict(checkpoint['run'])

        fetched_to = paths.PRETRAINED.path(dataset=run.dataset, model=run.model)
        path_in_repo = fetched_to.relative_to(paths.PRETRAINED_DIR).as_posix()
        replaces = file_exists(HF_REPO_ID, path_in_repo, repo_type='model')

        print(
            f'\nRun:     {run}\n'
            f'Step:    {checkpoint["step"]} '
            f'(selection score {checkpoint["selection_score"]:.4f})\n'
            f'Size:    {_mebibytes(model_path)}\n'
            f'Target:  {path_in_repo} on {HF_REPO_ID}\n'
            f'         {"replaces an existing file" if replaces else "adds a new file"}\n'
        )

        operations.append(CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=model_path))
        descriptions.append(
            f'- {run}: step {checkpoint["step"]}, '
            f'selection score {checkpoint["selection_score"]:.4f}'
        )

    print(f'Author:  {account}\n')
    prompt = (
        'Open a pull request?'
        if len(model_paths) == 1
        else f'Open a pull request for {len(model_paths)} models?'
    )
    if input(f'{prompt} [y/N] ').strip().lower() != 'y':
        raise SystemExit('Nothing was uploaded.')

    commit_message = (
        f'Publish {run}' if len(model_paths) == 1 else f'Publish {len(model_paths)} models'
    )
    commit = create_commit(
        repo_id=HF_REPO_ID,
        repo_type='model',
        operations=operations,
        commit_message=commit_message,
        commit_description='\n'.join(descriptions),
        create_pr=True,
    )

    print(f'Opened a pull request: {commit.pr_url}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Propose a run's best checkpoint as a published model, as a pull request "
        'against the Hugging Face model repo.'
    )
    parser.add_argument(
        '-m',
        '--checkpoint',
        type=paths.existing_file,
        metavar='CHECKPOINT',
        nargs='+',
        required=True,
        help='Path(s) to the checkpoint(s) to publish, from `outputs/checkpoints/best/`.',
    )
    args = parser.parse_args()

    run(args.checkpoint)


if __name__ == '__main__':
    main()
