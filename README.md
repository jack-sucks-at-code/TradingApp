# Market Making Hackathon Helper (Python + ML)

This project helps you run **one hackathon round at a time**:
1. Upload a stock's `train.csv`
2. Train and validate two models (`LinearRegression`, `XGBRegressor`)
3. Upload that stock's one-row `test.csv`
4. Generate market-making `Bid` and `Ask` quotes

It is designed for a first-year CS student who already knows Python.

---

## Why this helps in a Market Making competition

### What is a Market Maker?
A **market maker** continuously posts two prices:
- **Bid**: the price you're willing to buy at
- **Ask**: the price you're willing to sell at

You profit if you can buy near Bid and sell near Ask while controlling risk.

### Key terms
- **Predicted Price (Mid)**: your model's best estimate of fair value.
- **Spread**: `Ask - Bid`.
- **Tight spread**: smaller distance between Bid/Ask, more competitive, but can be riskier.
- **Wide spread**: safer buffer against model errors, but fewer fills.

### Using RMSE for uncertainty
- **RMSE** (Root Mean Squared Error) on validation data tells you how wrong your model tends to be.
- Larger RMSE means more uncertainty, so your spread should generally be wider.

---

## Quoting logic used in this script

After model training:
1. Pick the model with lower validation RMSE.
2. Predict the price for the single row in `test.csv`.
3. Compute quotes:

- `Bid = Prediction - (RMSE * risk_multiplier)`
- `Ask = Prediction + (RMSE * risk_multiplier)`

### Risk multiplier (your control knob)
In the script there is a `risk_multiplier` slider:
- `< 1.0` → aggressive/tighter spread
- `= 1.0` → balanced default (recommended)
- `> 1.0` → conservative/wider spread

The script also includes a safety check to ensure `Ask > Bid`.

---

## Data format expected

### Training CSV (`train.csv`)
- Many rows
- Feature columns like `col_0`, `col_1`, `col_2`, ...
- Must include target column: `target`

Example columns:
- `col_0, col_1, col_2, ..., target`

### Test CSV (`test.csv`)
- Exactly **one row**
- Contains only feature columns (no `target`)
- Must match feature columns used in training

---

## ML pipeline implemented

The script does the following:
1. Loads training data with pandas
2. Splits into train/validation with `train_test_split(test_size=0.2)`
3. Trains two models:
   - `LinearRegression`
   - `XGBRegressor`
4. Evaluates both with validation RMSE
5. Selects the better model (lower RMSE)
6. Predicts test-row target
7. Generates Bid/Ask quotes

---

## Installation

## 1) Create and activate a virtual environment (recommended)

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## 2) Install dependencies
```bash
pip install -r requirements.txt
```

---

## Run the app

### Recommended: Web GUI (super user-friendly)

Before running, configure app password (not committed to repo):

1. Create a local file `.streamlit/secrets.toml`
2. Add:

```toml
APP_PASSWORD = "your-private-key"
```

You can also set environment variable `APP_PASSWORD` instead.

```bash
streamlit run market_making_web.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

When the link opens, the first screen is a password prompt.

Web flow:
1. Upload **train.csv**
2. Click **Train + Validate Models**
3. Upload **test.csv** (exactly 1 row)
4. Adjust **Risk Multiplier** slider
5. Click **Generate Submission Quotes**

### Optional: Desktop Tkinter GUI

```bash
python market_making_gui.py
```

GUI flow:
1. Click **Upload Train CSV**
2. Click **Train + Validate Models**
3. Click **Upload Test CSV (1 row)**
4. Adjust **Risk Multiplier** slider
5. Click **Generate Submission Quotes**

Repeat this process for each of your 9 rounds/stocks.

---

## Output format

The app prints and displays:
- `Model Used: [Name]`
- `Predicted Price: [Value]`
- `Model Uncertainty (RMSE): [Value]`
- `--- SUBMISSION QUOTES ---`
- `Bid: [Value]`
- `Ask: [Value]`
- `Total Spread: [Ask - Bid]`

---

## Recommended competition workflow (9 rounds)

For each stock/round:
1. Use that stock's train file (e.g., `stock_3_train.csv`)
2. Train and inspect RMSE
3. Load its test file (e.g., `stock_3_test.csv`)
4. Start with risk multiplier = `1.0`
5. If market feels risky/volatile, increase multiplier (e.g., `1.2` to `1.8`)
6. If you need competitive fills, decrease multiplier carefully (e.g., `0.7` to `0.9`)

---

## Troubleshooting

- **Error: target column missing**
  - Make sure train CSV includes `target` exactly.

- **Error: test must have exactly one row**
  - Ensure your test file has a single observation.

- **Error: missing feature columns in test**
  - Train and test must use the same feature set.

- **XGBoost install issues**
  - Upgrade pip first: `pip install --upgrade pip`
  - Then reinstall: `pip install xgboost`

---

## Files

- `market_making_web.py` → Streamlit web GUI (recommended)
- `market_making_gui.py` → Tkinter desktop GUI
- `requirements.txt` → Python dependencies
- `README.md` → this guide

Good luck in the hackathon — the core idea is to treat RMSE like your uncertainty meter and adjust spread size with discipline.
