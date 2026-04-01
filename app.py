from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.agent.backtest import run_backtest_all_stocks
from src.agent.data import discover_stock_ids, load_all_stock_pairs
from src.agent.features import build_pair_train_arrays
from src.agent.model import evaluate_regressor, load_model, save_model, train_regressor


DEFAULT_DATA_DIR = "/Users/jackbew/Downloads/QMML-main/Hackathons/MarketMaking/hackathon_data"
DEFAULT_MODEL_PATH = "outputs/model.pkl"
DEFAULT_BACKTEST_OUT = "outputs/backtest_report.csv"
DEFAULT_PRED_OUT = "outputs/predictions.csv"


def validate_data(data_dir: str) -> tuple[int, list[int]]:
    stock_ids = discover_stock_ids(data_dir)
    _ = load_all_stock_pairs(data_dir)
    return len(stock_ids), stock_ids


def train_models(data_dir: str, model_out: str) -> pd.DataFrame:
    stock_pairs = load_all_stock_pairs(data_dir)
    model_bundle = {}
    metric_rows = []

    for pair in stock_pairs:
        x, y = build_pair_train_arrays(pair.train, pair.feature_columns)
        model = train_regressor(x, y)
        train_preds = model.predict(x)
        metrics = evaluate_regressor(y, train_preds)

        model_bundle[pair.stock_id] = {
            "model": model,
            "feature_columns": pair.feature_columns,
        }

        metric_rows.append(
            {
                "stock_id": pair.stock_id,
                "rmse": metrics.rmse,
                "mae": metrics.mae,
                "n_rows": len(pair.train),
                "n_features": len(pair.feature_columns),
            }
        )

    save_model(model_bundle, model_out)
    return pd.DataFrame(metric_rows).sort_values("stock_id").reset_index(drop=True)


def run_backtest(data_dir: str, model_path: str, risk_pct: float, fee_bps: float) -> pd.DataFrame:
    stock_pairs = load_all_stock_pairs(data_dir)
    model_bundle = load_model(model_path)
    report_df = run_backtest_all_stocks(
        stock_pairs=stock_pairs,
        model_bundle=model_bundle,
        risk_pct=risk_pct,
        fee_bps=fee_bps,
    )
    return report_df


def run_predict(data_dir: str, model_path: str) -> pd.DataFrame:
    stock_pairs = load_all_stock_pairs(data_dir)
    model_bundle = load_model(model_path)

    rows = []
    for pair in stock_pairs:
        stock_payload = model_bundle.get(pair.stock_id)
        if stock_payload is None:
            raise ValueError(f"No model found for stock {pair.stock_id}")

        model = stock_payload["model"]
        feature_columns = stock_payload["feature_columns"]
        x_test = pair.test[feature_columns].to_numpy()
        preds = model.predict(x_test)

        rows.append(
            pd.DataFrame(
                {
                    "stock_id": pair.stock_id,
                    "row_id": range(len(preds)),
                    "prediction": preds,
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


def _persist_csv(df: pd.DataFrame, output_path: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def main() -> None:
    st.set_page_config(page_title="Hackathon Trading Agent", layout="wide")
    st.title("Hackathon Trading Agent")
    st.caption("Train, backtest, and predict from CSV files with a simple risk percentage input.")

    with st.sidebar:
        st.header("Settings")
        data_dir = st.text_input("Data directory", value=DEFAULT_DATA_DIR)
        model_path = st.text_input("Model path", value=DEFAULT_MODEL_PATH)
        backtest_out = st.text_input("Backtest output CSV", value=DEFAULT_BACKTEST_OUT)
        pred_out = st.text_input("Prediction output CSV", value=DEFAULT_PRED_OUT)
        risk_pct = st.slider("Risk %", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        fee_bps = st.slider("Fee (bps)", min_value=0.0, max_value=20.0, value=2.0, step=0.5)

    validate_tab, train_tab, backtest_tab, predict_tab = st.tabs(
        ["Validate", "Train", "Backtest", "Predict"]
    )

    with validate_tab:
        st.subheader("1) Validate data")
        if st.button("Validate dataset", type="primary"):
            try:
                count, stock_ids = validate_data(data_dir)
                st.success(f"Validated {count} stock train/test pairs.")
                st.write({"stock_ids": stock_ids})
            except Exception as error:
                st.error(str(error))

    with train_tab:
        st.subheader("2) Train models")
        if st.button("Train and save model bundle", type="primary"):
            try:
                metrics_df = train_models(data_dir, model_path)
                st.success(f"Training complete. Model saved to: {model_path}")
                st.dataframe(metrics_df, use_container_width=True)
            except Exception as error:
                st.error(str(error))

    with backtest_tab:
        st.subheader("3) Backtest")
        if st.button("Run backtest", type="primary"):
            try:
                report_df = run_backtest(data_dir, model_path, risk_pct, fee_bps)
                saved = _persist_csv(report_df, backtest_out)
                st.success(f"Backtest complete. Saved to: {saved}")
                st.dataframe(report_df, use_container_width=True)
                st.download_button(
                    label="Download backtest CSV",
                    data=report_df.to_csv(index=False),
                    file_name="backtest_report.csv",
                    mime="text/csv",
                )
            except Exception as error:
                st.error(str(error))

    with predict_tab:
        st.subheader("4) Predict")
        if st.button("Generate predictions", type="primary"):
            try:
                pred_df = run_predict(data_dir, model_path)
                saved = _persist_csv(pred_df, pred_out)
                st.success(f"Predictions saved to: {saved}")
                st.dataframe(pred_df.head(100), use_container_width=True)
                st.download_button(
                    label="Download predictions CSV",
                    data=pred_df.to_csv(index=False),
                    file_name="predictions.csv",
                    mime="text/csv",
                )
            except Exception as error:
                st.error(str(error))


if __name__ == "__main__":
    main()
