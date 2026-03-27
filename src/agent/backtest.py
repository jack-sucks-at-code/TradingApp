from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .data import StockDataPair
from .metrics import summarize_backtest
from .policy import generate_positions


def run_backtest_for_stock(
    stock_pair: StockDataPair,
    model,
    risk_pct: float,
    fee_bps: float = 2.0,
) -> dict:
    train_df = stock_pair.train
    features = train_df[stock_pair.feature_columns].to_numpy()
    true_prices = train_df["target"].to_numpy()
    pred_prices = model.predict(features)

    if len(true_prices) < 2:
        raise ValueError(f"Not enough rows for backtest on stock {stock_pair.stock_id}")

    ref_prices = np.roll(true_prices, 1)
    ref_prices[0] = true_prices[0]

    positions = generate_positions(
        predicted_prices=pred_prices,
        reference_prices=ref_prices,
        risk_pct=risk_pct,
    )

    realized_returns = np.zeros_like(true_prices)
    realized_returns[1:] = true_prices[1:] / np.clip(true_prices[:-1], 1e-12, None) - 1.0

    turnover = np.abs(np.diff(positions, prepend=0.0))
    fee_rate = fee_bps / 10000.0
    strategy_returns = positions * realized_returns - turnover * fee_rate

    summary = asdict(summarize_backtest(strategy_returns))
    summary["stock_id"] = stock_pair.stock_id
    summary["avg_turnover"] = float(np.mean(turnover))
    return summary


def run_backtest_all_stocks(stock_pairs: list[StockDataPair], model_bundle: dict, risk_pct: float, fee_bps: float = 2.0) -> pd.DataFrame:
    rows = []
    for pair in stock_pairs:
        stock_payload = model_bundle.get(pair.stock_id)
        if stock_payload is None:
            raise ValueError(f"No model found for stock {pair.stock_id}")
        rows.append(
            run_backtest_for_stock(
                pair,
                model=stock_payload["model"],
                risk_pct=risk_pct,
                fee_bps=fee_bps,
            )
        )

    df = pd.DataFrame(rows).sort_values("stock_id").reset_index(drop=True)

    aggregate = {
        "stock_id": "ALL",
        "total_return": float(df["total_return"].mean()),
        "annualized_sharpe": float(df["annualized_sharpe"].mean()),
        "max_drawdown": float(df["max_drawdown"].mean()),
        "avg_turnover": float(df["avg_turnover"].mean()),
    }
    return pd.concat([df, pd.DataFrame([aggregate])], ignore_index=True)
