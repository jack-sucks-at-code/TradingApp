from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .backtest import run_backtest_all_stocks
from .data import load_all_stock_pairs
from .features import build_pair_train_arrays
from .model import evaluate_regressor, load_model, save_model, train_regressor


def cmd_validate_data(args: argparse.Namespace) -> None:
    stock_pairs = load_all_stock_pairs(args.data_dir)
    print(f"Validated {len(stock_pairs)} stock train/test pairs in {args.data_dir}")


def cmd_train(args: argparse.Namespace) -> None:
    stock_pairs = load_all_stock_pairs(args.data_dir)
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

    save_model(model_bundle, args.model_out)
    print(f"Saved model to: {args.model_out}")
    print(pd.DataFrame(metric_rows).sort_values("stock_id").to_string(index=False))


def cmd_backtest(args: argparse.Namespace) -> None:
    stock_pairs = load_all_stock_pairs(args.data_dir)
    model_bundle = load_model(args.model_path)
    report_df = run_backtest_all_stocks(
        stock_pairs=stock_pairs,
        model_bundle=model_bundle,
        risk_pct=args.risk_pct,
        fee_bps=args.fee_bps,
    )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output, index=False)
    print(report_df)
    print(f"Backtest report written to: {output}")


def cmd_predict(args: argparse.Namespace) -> None:
    stock_pairs = load_all_stock_pairs(args.data_dir)
    model_bundle = load_model(args.model_path)

    rows = []
    for pair in stock_pairs:
        stock_payload = model_bundle.get(pair.stock_id)
        if stock_payload is None:
            raise ValueError(f"No model found for stock {pair.stock_id}")

        model = stock_payload["model"]
        feature_columns = stock_payload["feature_columns"]
        x_test = pair.test[feature_columns].to_numpy()
        preds = model.predict(x_test)

        stock_pred = pd.DataFrame(
            {
                "stock_id": pair.stock_id,
                "row_id": range(len(preds)),
                "prediction": preds,
            }
        )
        rows.append(stock_pred)

    pred_df = pd.concat(rows, ignore_index=True)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output, index=False)
    print(f"Predictions saved to: {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hackathon trading agent CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-data", help="Validate CSV schema and file pairs")
    validate.add_argument("--data-dir", required=True)
    validate.set_defaults(func=cmd_validate_data)

    train = subparsers.add_parser("train", help="Train regression model")
    train.add_argument("--data-dir", required=True)
    train.add_argument("--model-out", default="outputs/model.pkl")
    train.set_defaults(func=cmd_train)

    backtest = subparsers.add_parser("backtest", help="Run risk-aware backtest")
    backtest.add_argument("--data-dir", required=True)
    backtest.add_argument("--model-path", default="outputs/model.pkl")
    backtest.add_argument("--risk-pct", type=float, default=1.0)
    backtest.add_argument("--fee-bps", type=float, default=2.0)
    backtest.add_argument("--out", default="outputs/backtest_report.csv")
    backtest.set_defaults(func=cmd_backtest)

    predict = subparsers.add_parser("predict", help="Predict prices for test rows")
    predict.add_argument("--data-dir", required=True)
    predict.add_argument("--model-path", default="outputs/model.pkl")
    predict.add_argument("--out", default="outputs/predictions.csv")
    predict.set_defaults(func=cmd_predict)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
