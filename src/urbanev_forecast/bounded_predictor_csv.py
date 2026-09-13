"""Bounded, identity-checked CSV prefixes for separately registered covariates."""
import hashlib
import io
from pathlib import Path
import numpy as np
import pandas as pd


def prefix_bytes(root, filename, rows, stage_limit, allowed_files):
    if (filename not in allowed_files or Path(filename).name != filename
            or isinstance(rows, bool) or not isinstance(rows, int)
            or not 1 <= rows <= stage_limit <= 1560):
        raise ValueError('Source or prefix outside registered stage')
    root = Path(root).resolve()
    path = (root/filename).resolve()
    if path.parent != root:
        raise ValueError('Source resolves outside registered root')
    with path.open('rb', buffering=0) as stream:
        lines = [stream.readline() for _ in range(rows+1)]
    if any(not line for line in lines):
        raise ValueError('Incomplete requested CSV prefix')
    return b''.join(lines)


def parse_prefix(payload, rows, columns, expected_sha256, nonnegative=False):
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError('Frozen prefix identity mismatch')
    frame = pd.read_csv(io.BytesIO(payload), index_col=0, parse_dates=True)
    if (frame.shape != (rows, len(columns)) or list(frame.columns.astype(str)) != list(columns)
            or not frame.index.equals(pd.date_range('2022-09-01', periods=rows, freq='h'))):
        raise ValueError('CSV columns or hourly timestamps do not match registration')
    values = frame.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (nonnegative and np.any(values < 0)):
        raise ValueError('Invalid values; no implicit imputation, clipping or row removal')
    return values, {'rows': rows, 'columns': len(columns), 'prefix_sha256': digest,
                    'numeric_row_range': [0, rows], 'imputation_performed': False,
                    'clipping_performed': False}
