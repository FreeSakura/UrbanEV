"""Small forecasting probes; none is claimed to be SOTA or a novel algorithm."""
from __future__ import annotations
import math
from types import SimpleNamespace
from pathlib import Path
import sys
import torch
from torch import nn


class SeasonalForecaster(nn.Module):
    def __init__(self, history=168, horizon=12, channels=275, variant="seasonal_linear", width=64):
        super().__init__()
        if history <= 24 or not 1 <= horizon <= 24 or channels < 1 or width < 1:
            raise ValueError("Require history>24, horizon in 1..24 and positive channels/width")
        choices = ("seasonal_linear", "seasonal_mlp", "innovation_attention", "level_attention")
        if variant not in choices:
            raise ValueError(f"variant must be one of {choices}")
        self.history, self.horizon, self.channels, self.variant = history, horizon, channels, variant
        features = history-24
        if variant == "seasonal_linear":
            self.head = nn.Linear(features, horizon)
        else:
            self.encoder = nn.Sequential(nn.Linear(features, width), nn.GELU())
            self.head = nn.Linear(width, horizon)
        if variant.endswith("attention"):
            self.query, self.key = nn.Linear(width, 16, bias=False), nn.Linear(width, 16, bias=False)
            self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.history, self.channels):
            raise ValueError("Expected [batch, history, channels]")
        anchor = x[:, -24:-24+self.horizon] if self.horizon < 24 else x[:, -24:]
        features = (x[:,24:] if self.variant == "level_attention" else x[:,24:]-x[:,:-24]).transpose(1,2)
        if self.variant == "seasonal_linear":
            residual = self.head(features)
        else:
            hidden = self.encoder(features)
            if self.variant.endswith("attention") and self.channels > 1:
                logits = self.query(hidden) @ self.key(hidden).transpose(-1,-2) / math.sqrt(16)
                diagonal = torch.eye(self.channels, dtype=torch.bool, device=x.device)
                weights = logits.masked_fill(diagonal, -torch.inf).softmax(dim=-1)
                hidden = hidden + torch.tanh(self.gate) * (weights @ hidden)
            residual = self.head(hidden)
        return anchor + residual.transpose(1,2)


def build_model(name, history, horizon, channels, width=64):
    if name != "timexer":
        return SeasonalForecaster(history, horizon, channels, name, width)
    # Reuse the attributed upstream snapshot without editing its implementation.
    source = Path(__file__).resolve().parents[2] / "models/timexer"
    sys.path.insert(0, str(source))
    from TimeXer import Model
    config = SimpleNamespace(task_name="long_term_forecast", features="M", seq_len=history,
        pred_len=horizon, use_norm=True, patch_len=12, d_model=width, n_heads=4,
        e_layers=2, d_ff=2*width, dropout=.1, enc_in=channels, factor=3,
        activation="gelu", embed="timeF", freq="h")
    if history % 12 or width % 4:
        raise ValueError("TimeXer requires history divisible by 12 and width divisible by 4")
    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Model(config)
        def forward(self, x):
            return self.model(x, None, None, None)
    return Adapter()
