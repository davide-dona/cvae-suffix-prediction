"""Hydra output records shared by independently runnable pipeline stages."""

import json
import platform
import subprocess
import sys
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


def output_path(name: str) -> Path:
    """Return a path inside the active Hydra invocation."""
    path = Path(HydraConfig.get().runtime.output_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def start_stage(config: DictConfig) -> None:
    """Reserve the output directory and persist the resolved configuration."""
    with output_path('invocation.json').open('x') as file:
        revision = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False
        ).stdout.strip()
        dirty = subprocess.run(
            ['git', 'status', '--porcelain'], capture_output=True, text=True, check=False
        ).stdout.strip()
        json.dump(
            {
                'revision': revision,
                'dirty': bool(dirty),
                'argv': sys.argv,
                'python': platform.python_version(),
            },
            file,
            indent=2,
        )
    save_config(config)


def save_config(config: DictConfig) -> None:
    """Save all effective settings, including resolved checkpoint-derived values."""
    OmegaConf.save(config=config, f=output_path('config.yaml'), resolve=True)
