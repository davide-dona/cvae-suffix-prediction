import argparse
from pathlib import Path


def existing_file(value: str) -> Path:
    """Read a path argument that has to already exist.

    Args:
        value: What the flag was given.
    Returns:
        That path.
    Raises:
        argparse.ArgumentTypeError: If nothing is there, so a mistyped path fails while the
            arguments are being read rather than part-way into whatever reads the file, and as
            argparse's own usage error rather than as a traceback.
    """
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f'no such file: {path}')
    return path


def existing_directory(value: str) -> Path:
    """Read a path argument that has to already be a directory.

    Args:
        value: What the flag was given.
    Returns:
        That path.
    Raises:
        argparse.ArgumentTypeError: If nothing is there or a file is, so a mistyped path fails
            while the arguments are being read rather than part-way into whatever walks it, and as
            argparse's own usage error rather than as a traceback.
    """
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f'no such directory: {path}')
    return path
