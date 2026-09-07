"""Portable artifact provenance and Arrow metadata."""

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def with_metadata(schema: pa.Schema, metadata: dict[str, str]) -> pa.Schema:
    return schema.with_metadata(
        (schema.metadata or {}) | {b'provenance': json.dumps(metadata).encode()}
    )


def read_metadata(parquet: pq.ParquetFile) -> dict[str, str]:
    raw = (parquet.schema_arrow.metadata or {}).get(b'provenance')
    if raw is None:
        raise ValueError('Missing artifact provenance; regenerate this file.')
    metadata = json.loads(raw)
    if not isinstance(metadata, dict) or not all(
        isinstance(metadata.get(key), str) and metadata[key]
        for key in ('dataset', 'model', 'checkpoint_sha256')
    ):
        raise ValueError('Invalid artifact provenance.')
    return metadata


def with_vocabulary(schema: pa.Schema, vocabulary: Sequence[str]) -> pa.Schema:
    return schema.with_metadata(
        (schema.metadata or {}) | {b'activities': json.dumps(list(vocabulary)).encode()}
    )


def read_vocabulary(schema: pa.Schema) -> tuple[str, ...]:
    raw = (schema.metadata or {}).get(b'activities')
    if raw is None:
        raise ValueError('Missing activity vocabulary; regenerate this file.')
    return tuple(json.loads(raw))


def group_by_model(artifacts: Iterable[tuple[dict[str, str], Path]]) -> dict[str, dict[str, Path]]:
    grouped: dict[str, dict[str, Path]] = {}
    for metadata, path in artifacts:
        dataset, model = metadata['dataset'], metadata['model']
        models = grouped.setdefault(dataset, {})
        if model in models:
            raise ValueError(f'{dataset} has two reports for {model}: {models[model]}, {path}')
        models[model] = path
    return grouped
