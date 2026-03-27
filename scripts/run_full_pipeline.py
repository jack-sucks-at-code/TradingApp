from __future__ import annotations

import argparse
import subprocess
import sys


def run(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full training + backtest + predict pipeline")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--risk-pct", type=float, default=1.0)
    parser.add_argument("--model-path", default="outputs/model.pkl")
    parser.add_argument("--backtest-out", default="outputs/backtest_report.csv")
    parser.add_argument("--pred-out", default="outputs/predictions.csv")
    args = parser.parse_args()

    run([sys.executable, "-m", "src.agent.cli", "validate-data", "--data-dir", args.data_dir])
    run([
        sys.executable,
        "-m",
        "src.agent.cli",
        "train",
        "--data-dir",
        args.data_dir,
        "--model-out",
        args.model_path,
    ])
    run([
        sys.executable,
        "-m",
        "src.agent.cli",
        "backtest",
        "--data-dir",
        args.data_dir,
        "--model-path",
        args.model_path,
        "--risk-pct",
        str(args.risk_pct),
        "--out",
        args.backtest_out,
    ])
    run([
        sys.executable,
        "-m",
        "src.agent.cli",
        "predict",
        "--data-dir",
        args.data_dir,
        "--model-path",
        args.model_path,
        "--out",
        args.pred_out,
    ])


if __name__ == "__main__":
    main()
