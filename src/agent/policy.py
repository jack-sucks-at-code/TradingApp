from __future__ import annotations

import numpy as np


def generate_positions(
    predicted_prices: np.ndarray,
    reference_prices: np.ndarray,
    risk_pct: float,
    edge_threshold: float = 0.001,
) -> np.ndarray:
    if risk_pct <= 0:
        raise ValueError("risk_pct must be > 0")

    if len(predicted_prices) != len(reference_prices):
        raise ValueError("predicted_prices and reference_prices must have same length")

    expected_return = (predicted_prices - reference_prices) / np.clip(reference_prices, 1e-8, None)

    signal_strength = np.zeros_like(expected_return)
    active = np.abs(expected_return) >= edge_threshold
    signal_strength[active] = np.sign(expected_return[active]) * np.minimum(
        np.abs(expected_return[active]) / edge_threshold,
        1.0,
    )

    position_scale = risk_pct / 100.0
    positions = signal_strength * position_scale
    return np.clip(positions, -position_scale, position_scale)
