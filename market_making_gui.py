"""
Market Making Helper (Hackathon Edition)

This script provides a beginner-friendly GUI workflow to:
1) Upload a training CSV
2) Train two models (Linear Regression and XGBoost Regressor)
3) Compare validation RMSE and choose the better model
4) Upload a one-row test CSV
5) Predict price and generate Bid/Ask quotes

Designed for one round at a time (repeat for 9 rounds/stocks).
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# -----------------------------------------------------------------------------
# Risk management setting
# Set to < 1.0 for aggressive market making (tight spread),
# and > 1.0 for conservative/safe trading (wide spread).
# -----------------------------------------------------------------------------
DEFAULT_RISK_MULTIPLIER = 1.0
EPSILON_SPREAD = 1e-9
RANDOM_STATE = 42
TARGET_COLUMN = "target"


def load_csv(file_path: str | Path) -> pd.DataFrame:
    """Load CSV into a pandas DataFrame with basic file checks."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got: {path.suffix}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")
    return df


def validate_train_df(df: pd.DataFrame, target_col: str) -> None:
    """Validate training data shape and required target column."""
    if target_col not in df.columns:
        raise ValueError(
            f"Training CSV must include target column '{target_col}'. "
            f"Found columns: {list(df.columns)}"
        )

    if df[target_col].isna().any():
        raise ValueError("Target column contains missing values. Please clean your data.")


def prepare_features_and_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features (X) and target (y), then apply minimal cleaning.

    Why this cleaning?
    - We keep only numeric columns so both models can train safely.
    - We fill missing feature values with median for stability.
    """
    y = df[target_col]
    X = df.drop(columns=[target_col])

    X = X.select_dtypes(include=[np.number]).copy()
    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns found after removing target.")

    X = X.fillna(X.median(numeric_only=True))

    if y.dtype.kind not in "biufc":
        raise ValueError("Target column must be numeric for regression.")

    return X, y


def prepare_test_features(test_df: pd.DataFrame, train_feature_cols: pd.Index) -> pd.DataFrame:
    """
    Prepare test features to match training schema exactly.

    Expectations from your hackathon format:
    - test.csv has exactly one row
    - test.csv contains only feature columns (no target)
    """
    if test_df.shape[0] != 1:
        raise ValueError(f"Test CSV must contain exactly one row. Found {test_df.shape[0]} rows.")

    missing_cols = [col for col in train_feature_cols if col not in test_df.columns]
    if missing_cols:
        raise ValueError(
            "Test CSV is missing required feature columns: "
            f"{missing_cols[:10]}{'...' if len(missing_cols) > 10 else ''}"
        )

    X_test = test_df[train_feature_cols].copy()
    X_test = X_test.select_dtypes(include=[np.number])

    if list(X_test.columns) != list(train_feature_cols):
        # If select_dtypes dropped some columns, detect and explain clearly.
        dropped = [c for c in train_feature_cols if c not in X_test.columns]
        raise ValueError(
            "Some required feature columns in test data are non-numeric or missing after type filtering: "
            f"{dropped}"
        )

    X_test = X_test.fillna(X_test.median(numeric_only=True))
    return X_test


def train_and_evaluate_models(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[Dict[str, object], Dict[str, float], str]:
    """
    Train LinearRegression and XGBRegressor using an 80/20 split.

    Returns:
    - trained models dict
    - rmse scores dict
    - name of best model (lowest RMSE)
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    models: Dict[str, object] = {
        "LinearRegression": LinearRegression(),
        "XGBRegressor": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=4,
        ),
    }

    rmse_scores: Dict[str, float] = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, preds)
        rmse_scores[model_name] = float(rmse)

    best_model_name = min(rmse_scores, key=rmse_scores.get)
    return models, rmse_scores, best_model_name


def generate_quotes(prediction: float, rmse: float, risk_multiplier: float) -> Tuple[float, float, float]:
    """
    Generate market-making quotes from model prediction and uncertainty.

    Custom volatility-adjusted spread logic:
    Bid = Prediction - (RMSE * risk_multiplier)
    Ask = Prediction + (RMSE * risk_multiplier)

    Safety check:
    Ensure Ask is always greater than Bid.
    """
    spread_component = rmse * risk_multiplier
    bid = prediction - spread_component
    ask = prediction + spread_component

    if ask <= bid:
        ask = bid + EPSILON_SPREAD

    total_spread = ask - bid
    return bid, ask, total_spread


class MarketMakingApp:
    """Simple Tkinter GUI for one-round-at-a-time hackathon workflow."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Market Making Helper")
        self.root.geometry("760x620")

        self.train_path: Optional[Path] = None
        self.test_path: Optional[Path] = None

        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

        self.models: Dict[str, object] = {}
        self.rmse_scores: Dict[str, float] = {}
        self.best_model_name: Optional[str] = None

        self.feature_cols: Optional[pd.Index] = None

        self.risk_multiplier = tk.DoubleVar(value=DEFAULT_RISK_MULTIPLIER)
        self.target_col_var = tk.StringVar(value=TARGET_COLUMN)

        self._build_ui()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=16)
        main.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(
            main,
            text="Market Making Hackathon Helper",
            font=("Arial", 16, "bold"),
        )
        title.pack(anchor="w", pady=(0, 10))

        desc = ttk.Label(
            main,
            text=(
                "Flow: 1) Upload train CSV → 2) Train models → "
                "3) Upload one-row test CSV → 4) Generate Bid/Ask"
            ),
        )
        desc.pack(anchor="w", pady=(0, 12))

        target_frame = ttk.LabelFrame(main, text="Settings", padding=12)
        target_frame.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(target_frame, text="Target column name:").grid(row=0, column=0, sticky="w")
        ttk.Entry(target_frame, textvariable=self.target_col_var, width=24).grid(
            row=0, column=1, sticky="w", padx=(8, 20)
        )

        ttk.Label(target_frame, text="Risk Multiplier:").grid(row=1, column=0, sticky="w", pady=(12, 0))

        slider = ttk.Scale(
            target_frame,
            from_=0.1,
            to=3.0,
            variable=self.risk_multiplier,
            orient="horizontal",
            length=250,
            command=self._on_slider_change,
        )
        slider.grid(row=1, column=1, sticky="w", padx=(8, 12), pady=(12, 0))

        self.multiplier_label = ttk.Label(target_frame, text=f"{DEFAULT_RISK_MULTIPLIER:.2f} (recommended)")
        self.multiplier_label.grid(row=1, column=2, sticky="w", pady=(12, 0))

        ttk.Label(
            target_frame,
            text="< 1.0 = tighter/aggressive quotes | > 1.0 = wider/conservative quotes",
            foreground="#444444",
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))

        actions = ttk.LabelFrame(main, text="Round Workflow", padding=12)
        actions.pack(fill=tk.X, pady=(0, 12))

        self.btn_load_train = ttk.Button(actions, text="1) Upload Train CSV", command=self.load_train_file)
        self.btn_load_train.grid(row=0, column=0, padx=(0, 10), pady=4, sticky="w")

        self.btn_train = ttk.Button(
            actions,
            text="2) Train + Validate Models",
            command=self.train_models,
            state=tk.DISABLED,
        )
        self.btn_train.grid(row=0, column=1, padx=(0, 10), pady=4, sticky="w")

        self.btn_load_test = ttk.Button(
            actions,
            text="3) Upload Test CSV (1 row)",
            command=self.load_test_file,
            state=tk.DISABLED,
        )
        self.btn_load_test.grid(row=0, column=2, padx=(0, 10), pady=4, sticky="w")

        self.btn_quote = ttk.Button(
            actions,
            text="4) Generate Submission Quotes",
            command=self.generate_submission_quotes,
            state=tk.DISABLED,
        )
        self.btn_quote.grid(row=0, column=3, pady=4, sticky="w")

        self.status_label = ttk.Label(main, text="Status: Waiting for training file upload.")
        self.status_label.pack(anchor="w", pady=(0, 8))

        self.output_text = tk.Text(main, height=22, width=95)
        self.output_text.pack(fill=tk.BOTH, expand=True)

        self._log("Welcome! Upload a train CSV to begin round 1.")

    def _on_slider_change(self, _event: str) -> None:
        self.multiplier_label.config(text=f"{self.risk_multiplier.get():.2f} (recommended: 1.00)")

    def _log(self, message: str) -> None:
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)

    def _set_status(self, message: str) -> None:
        self.status_label.config(text=f"Status: {message}")

    def load_train_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select Train CSV",
            filetypes=[("CSV files", "*.csv")],
        )
        if not file_path:
            return

        try:
            self.train_df = load_csv(file_path)
            self.train_path = Path(file_path)
            target_col = self.target_col_var.get().strip()
            validate_train_df(self.train_df, target_col)

            self._set_status(f"Train file loaded: {self.train_path.name}")
            self._log(f"Loaded train CSV: {self.train_path}")
            self.btn_train.config(state=tk.NORMAL)

            self.test_df = None
            self.test_path = None
            self.models = {}
            self.rmse_scores = {}
            self.best_model_name = None
            self.feature_cols = None
            self.btn_load_test.config(state=tk.DISABLED)
            self.btn_quote.config(state=tk.DISABLED)

        except Exception as exc:
            messagebox.showerror("Train File Error", str(exc))
            self._log(f"Train file error: {exc}")

    def train_models(self) -> None:
        if self.train_df is None:
            messagebox.showwarning("Missing data", "Please upload a train CSV first.")
            return

        target_col = self.target_col_var.get().strip()

        try:
            validate_train_df(self.train_df, target_col)
            X, y = prepare_features_and_target(self.train_df, target_col)
            self.feature_cols = X.columns

            self.models, self.rmse_scores, self.best_model_name = train_and_evaluate_models(X, y)

            self._set_status("Models trained. Upload test CSV to generate quotes.")
            self.btn_load_test.config(state=tk.NORMAL)

            self._log("\n=== VALIDATION RESULTS ===")
            self._log(f"LinearRegression RMSE: {self.rmse_scores['LinearRegression']:.6f}")
            self._log(f"XGBRegressor RMSE: {self.rmse_scores['XGBRegressor']:.6f}")
            self._log(f"Best model selected (lowest RMSE): {self.best_model_name}")

        except Exception as exc:
            messagebox.showerror("Training Error", str(exc))
            self._log(f"Training error: {exc}")

    def load_test_file(self) -> None:
        if self.best_model_name is None:
            messagebox.showwarning("Train first", "Please train models before uploading test file.")
            return

        file_path = filedialog.askopenfilename(
            title="Select Test CSV",
            filetypes=[("CSV files", "*.csv")],
        )
        if not file_path:
            return

        try:
            self.test_df = load_csv(file_path)
            self.test_path = Path(file_path)

            if self.feature_cols is None:
                raise ValueError("Feature schema is unavailable. Re-run training.")

            _ = prepare_test_features(self.test_df, self.feature_cols)

            self._set_status(f"Test file loaded: {self.test_path.name}")
            self._log(f"Loaded test CSV: {self.test_path}")
            self.btn_quote.config(state=tk.NORMAL)

        except Exception as exc:
            messagebox.showerror("Test File Error", str(exc))
            self._log(f"Test file error: {exc}")

    def generate_submission_quotes(self) -> None:
        if self.best_model_name is None or self.test_df is None:
            messagebox.showwarning("Missing step", "Train models and upload test CSV first.")
            return

        risk_multiplier = float(self.risk_multiplier.get())
        if risk_multiplier <= 0:
            messagebox.showerror("Invalid multiplier", "Risk multiplier must be greater than 0.")
            return

        try:
            if self.feature_cols is None:
                raise ValueError("Feature schema unavailable. Re-run training.")

            X_test = prepare_test_features(self.test_df, self.feature_cols)

            best_model = self.models[self.best_model_name]
            prediction = float(best_model.predict(X_test)[0])
            rmse = float(self.rmse_scores[self.best_model_name])

            bid, ask, total_spread = generate_quotes(prediction, rmse, risk_multiplier)

            output_lines = [
                "",
                "=== ROUND QUOTE OUTPUT ===",
                f"Model Used: {self.best_model_name}",
                f"Predicted Price: {prediction:.6f}",
                f"Model Uncertainty (RMSE): {rmse:.6f}",
                "--- SUBMISSION QUOTES ---",
                f"Bid: {bid:.6f}",
                f"Ask: {ask:.6f}",
                f"Total Spread: {total_spread:.6f}",
            ]

            for line in output_lines:
                print(line)
                self._log(line)

            self._set_status("Quotes generated. Adjust risk multiplier and regenerate if needed.")

        except Exception as exc:
            messagebox.showerror("Quote Generation Error", str(exc))
            self._log(f"Quote generation error: {exc}")


def main() -> None:
    root = tk.Tk()
    style = ttk.Style()

    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    app = MarketMakingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
