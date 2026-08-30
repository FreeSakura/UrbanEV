from __future__ import annotations

import numpy as np


def official_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """Byte-for-byte semantic equivalent of the repository metric implementation."""
    eps = 2e-2
    m_true = np.asarray(true, dtype=float).copy()
    m_pred = np.asarray(pred, dtype=float).copy()
    m_true[np.where(m_true <= eps)] = np.abs(m_true[np.where(m_true <= eps)]) + eps
    m_pred[np.where(m_true <= eps)] = np.abs(m_pred[np.where(m_true <= eps)]) + eps
    return _finish(m_pred, m_true, np.asarray(pred, dtype=float), np.asarray(true, dtype=float))


def audited_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    """Correct near-zero MAPE masking and compute RAE on the unmodified arrays."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    mask = true <= 2e-2
    safe_true = true.copy()
    safe_true[mask] = np.abs(safe_true[mask]) + 2e-2
    safe_pred = pred.copy()
    safe_pred[mask] = np.abs(safe_pred[mask]) + 2e-2
    err = np.abs(safe_pred - safe_true)
    denominator = np.sum(np.abs(true - np.mean(true)))
    return {
        "MSE": float(np.mean((pred - true) ** 2)),
        "RMSE": float(np.sqrt(np.mean((pred - true) ** 2))),
        "MAPE": float(np.mean(err / np.abs(safe_true))),
        "RAE": float(np.sum(np.abs(pred - true)) / denominator) if denominator else float("nan"),
        "MAE": float(np.mean(np.abs(pred - true))),
    }


def _finish(m_pred: np.ndarray, m_true: np.ndarray, pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    mse = np.mean((pred - true) ** 2)
    denominator = np.sum(np.abs(np.mean(m_true) - m_true))
    return {
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAPE": float(np.mean(np.abs((m_pred - m_true) / m_true))),
        "RAE": float(np.sum(np.abs(m_pred - m_true)) / denominator) if denominator else float("nan"),
        "MAE": float(np.mean(np.abs(pred - true))),
    }

