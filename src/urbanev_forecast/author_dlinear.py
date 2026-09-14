"""Verified, external author DLinear; no local replacement of its model core."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

REVISION = "0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6"
MODEL_SHA256 = "0893b53cb6473d6bdca7aeca514cb3ee12efa6df227c29c4469571c9711451cc"


def load_author_model(source_root: Path):
    """Verify exact downloaded bytes before importing; no network or path mutation."""
    path = Path(source_root) / "models/DLinear.py"
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != MODEL_SHA256:
        raise ValueError("author DLinear source hash mismatch")
    spec = importlib.util.spec_from_file_location(f"_author_dlinear_{REVISION}", path)
    module = importlib.util.module_from_spec(spec)
    # Execute precisely the already-verified bytes, avoiding a stale pyc or reread.
    exec(compile(payload, str(path), "exec"), module.__dict__)
    return module.Model


class DLinearAdapter(nn.Module):
    """O-only [batch,L,zone] -> [batch,H,zone], with shared temporal weights.

    The author's shared mode is channel-independent: passing all zones does not
    create spatial interaction or allow duration to influence occupancy outputs.
    External preprocessing and training selection belong to the future protocol.
    """
    def __init__(self, source_root: Path, *, history: int, horizon: int, channels: int = 275):
        super().__init__()
        for name, value in (("history", history), ("horizon", horizon), ("channels", channels)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.history, self.horizon, self.channels = history, horizon, channels
        author = load_author_model(source_root)
        self.model = author(SimpleNamespace(seq_len=history, pred_len=horizon, enc_in=channels, individual=False))

    def forward(self, occupancy_history):
        x = occupancy_history
        if x.ndim != 3 or x.shape[0] < 1 or x.shape[1:] != (self.history, self.channels):
            raise ValueError("expected nonempty [batch,history,zone] occupancy tensor")
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("occupancy input must be finite floating point")
        result = self.model(x)
        if result.shape != (x.shape[0], self.horizon, self.channels):
            raise ValueError("unexpected author output shape")
        return result
