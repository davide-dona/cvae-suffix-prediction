"""Sequential CPU evaluation of generated artifacts."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.artifacts import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--roots', nargs='+', default=['outputs', 'pinned'])
    parser.add_argument('overrides', nargs='*', help='Hydra evaluation overrides')
    args = parser.parse_args()
    roots = [Path(root) for root in args.roots]
    completed = set()
    for root in roots:
        for report in root.rglob('evaluation.json'):
            data = json.loads(report.read_text())
            if report.with_name('prefix_scores.parquet').is_file():
                completed.add(data['metadata']['source_sha256'])
    files = sorted({file for root in roots for file in root.rglob('generations.parquet')})
    if not files:
        raise SystemExit('No generations.parquet files found')
    failed = []
    for file in files:
        digest = sha256(file)
        if digest in completed and not args.force:
            print(f'skip {file}', flush=True)
            continue
        status = subprocess.run(
            [
                sys.executable,
                '-m',
                'pipelines.evaluate',
                f'generations={json.dumps(str(file.resolve()))}',
                *args.overrides,
            ],
            check=False,
        )
        if status.returncode:
            failed.append(file)
        else:
            completed.add(digest)
    if failed:
        raise SystemExit(f'Evaluation failed for: {failed}')


if __name__ == '__main__':
    main()
