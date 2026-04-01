from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BacktestMetrics:
    total_return: float
    annualized_sharpe: float
    max_drawdown: float


def compute_max_drawdown(equity_curve: np.ndarray) -> float:
    running_peak = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve / np.clip(running_peak, 1e-12, None) - 1.0
    return float(np.min(drawdown))


def compute_annualized_sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    if len(returns) < 2:
        return 0.0

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns) + 1e-12)
    return float((mean_ret / std_ret) * np.sqrt(periods_per_year))


def summarize_backtest(returns: np.ndarray) -> BacktestMetrics:
    equity_curve = np.cumprod(1.0 + returns)
    return BacktestMetrics(
        total_return=float(equity_curve[-1] - 1.0),
        annualized_sharpe=compute_annualized_sharpe(returns),
        max_drawdown=compute_max_drawdown(equity_curve),
    )
