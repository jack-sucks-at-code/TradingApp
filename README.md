# Hackathon Trading Agent (CSV + Risk Input)

This project trains a price prediction model from CSV files and turns predictions into trading actions using an input risk percentage.

## Expected data format

- Train files: `stock_<id>_train.csv` with feature columns (`col_*`) plus `target`
- Test files: `stock_<id>_test.csv` with the same feature columns as that stock's train file

Example dataset folder:

```
/path/to/hackathon_data/
  stock_1_train.csv
  stock_1_test.csv
  ...
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Commands

Validate data:

```bash
python -m src.agent.cli validate-data --data-dir "/path/to/hackathon_data"
```

Train model:

```bash
python -m src.agent.cli train --data-dir "/path/to/hackathon_data" --model-out outputs/model.pkl
```

Backtest with risk percentage:

```bash
python -m src.agent.cli backtest --data-dir "/path/to/hackathon_data" --risk-pct 1.0 --model-path outputs/model.pkl
```

Predict on test files:

```bash
python -m src.agent.cli predict --data-dir "/path/to/hackathon_data" --model-path outputs/model.pkl --out outputs/predictions.csv
```

Run full pipeline:

```bash
python scripts/run_full_pipeline.py --data-dir "/path/to/hackathon_data" --risk-pct 1.0
```

## GUI (recommended)

Launch the user-friendly app:

```bash
streamlit run app.py
```

In the GUI you can:

- Validate dataset schema
- Train models
- Run backtests with input risk percentage
- Generate and download predictions

## Notes

- `risk-pct` is the position-risk scalar. Higher value means larger position sizes.
- This is a hackathon baseline: prediction + risk sizing + backtest.
