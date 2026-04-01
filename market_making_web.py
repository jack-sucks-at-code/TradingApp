"""
Market Making Web GUI (Streamlit)

Beginner-friendly browser app for one-round-at-a-time market making:
1) Upload train CSV
2) Train LinearRegression and XGBRegressor with 80/20 validation split
3) Upload one-row test CSV
4) Generate Bid/Ask quotes using RMSE-based uncertainty spread
"""

from __future__ import annotations

import os
from io import StringIO
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
    XGBOOST_IMPORT_ERROR = ""
except Exception as xgb_exc:
    XGBRegressor = None
    XGBOOST_AVAILABLE = False
    XGBOOST_IMPORT_ERROR = str(xgb_exc)

TARGET_COLUMN_DEFAULT = "target"
DEFAULT_RISK_MULTIPLIER = 1.0
RANDOM_STATE = 42
EPSILON_SPREAD = 1e-9


def get_app_password() -> str:
    """
    Get app password from a non-committed source.

    Priority:
    1) Streamlit secrets: APP_PASSWORD
    2) Environment variable: APP_PASSWORD
    """
    password_from_secrets = st.secrets.get("APP_PASSWORD")
    if password_from_secrets:
        return str(password_from_secrets)

    return os.getenv("APP_PASSWORD", "")


def password_gate() -> bool:
    """Render the first screen as a password prompt and return auth status."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if st.session_state["authenticated"]:
        return True

    st.title("🔐 Secure Access")
    st.write("Enter password to access the trading app.")

    configured_password = get_app_password()
    if not configured_password:
        st.error(
            "No app password configured. Set APP_PASSWORD in .streamlit/secrets.toml "
            "or as an environment variable."
        )
        return False

    entered_password = st.text_input("Password", type="password", key="password_input")
    if st.button("Unlock", use_container_width=True):
        if entered_password == configured_password:
            st.session_state["authenticated"] = True
            st.success("Access granted.")
            st.rerun()
        else:
            st.error("Invalid password.")

    return False


def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Load Streamlit UploadedFile as DataFrame with clear validation."""
    if uploaded_file is None:
        raise ValueError("No file uploaded.")

    content = uploaded_file.getvalue().decode("utf-8")
    df = pd.read_csv(StringIO(content))

    if df.empty:
        raise ValueError("Uploaded CSV is empty.")

    return df


def validate_train_df(df: pd.DataFrame, target_col: str) -> None:
    if target_col not in df.columns:
        raise ValueError(
            f"Training CSV must include target column '{target_col}'. "
            f"Found columns: {list(df.columns)}"
        )

    if df[target_col].isna().any():
        raise ValueError("Target column contains missing values.")

    if not np.issubdtype(df[target_col].dtype, np.number):
        raise ValueError("Target column must be numeric.")


def prepare_features_and_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build model-ready train features/target.

    Why numeric-only features?
    - Keeps the pipeline simple and stable for hackathon speed.
    - Avoids encoding complexity when data already uses col_0, col_1 style numeric features.
    """
    y = df[target_col]
    X = df.drop(columns=[target_col])

    X = X.select_dtypes(include=[np.number]).copy()
    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns found after removing target.")

    X = X.fillna(X.median(numeric_only=True))
    return X, y


def train_and_evaluate_models(X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, object], Dict[str, float], str]:
    """Train both models and compare validation RMSE."""
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    models: Dict[str, object] = {
        "LinearRegression": LinearRegression(),
    }

    if XGBOOST_AVAILABLE and XGBRegressor is not None:
        models["XGBRegressor"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=4,
        )

    rmse_scores: Dict[str, float] = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse_scores[model_name] = float(root_mean_squared_error(y_val, preds))

    best_model_name = min(rmse_scores, key=rmse_scores.get)
    return models, rmse_scores, best_model_name


def prepare_test_features(test_df: pd.DataFrame, train_feature_cols: pd.Index) -> pd.DataFrame:
    """Validate test schema and align with training feature order."""
    if test_df.shape[0] != 1:
        raise ValueError(f"Test CSV must contain exactly one row. Found {test_df.shape[0]} rows.")

    missing_cols = [col for col in train_feature_cols if col not in test_df.columns]
    if missing_cols:
        raise ValueError(f"Test CSV is missing required feature columns: {missing_cols}")

    X_test = test_df[train_feature_cols].copy()
    X_test = X_test.select_dtypes(include=[np.number])

    if list(X_test.columns) != list(train_feature_cols):
        dropped = [col for col in train_feature_cols if col not in X_test.columns]
        raise ValueError(
            f"Some required columns are non-numeric or dropped by cleaning: {dropped}"
        )

    X_test = X_test.fillna(X_test.median(numeric_only=True))
    return X_test


def generate_quotes(prediction: float, rmse: float, risk_multiplier: float) -> Tuple[float, float, float]:
    """
    Custom volatility-adjusted spread:
    Bid = Prediction - (RMSE * risk_multiplier)
    Ask = Prediction + (RMSE * risk_multiplier)

    risk_multiplier comment for risk management:
    - Set < 1.0 for aggressive market making (tight spread)
    - Set > 1.0 for conservative/safe trading (wide spread)
    """
    spread_component = rmse * risk_multiplier
    bid = prediction - spread_component
    ask = prediction + spread_component

    if ask <= bid:
        ask = bid + EPSILON_SPREAD

    total_spread = ask - bid
    return bid, ask, total_spread


def main() -> None:
    st.set_page_config(page_title="Market Making Web GUI", page_icon="📈", layout="centered")

    if not password_gate():
        return

    st.title("📈 Market Making Hackathon Web GUI")
    st.caption("One round at a time: Upload train.csv → train models → upload test.csv → generate quotes")

    if not XGBOOST_AVAILABLE:
        st.warning(
            "XGBoost is unavailable in this environment, so the app will use LinearRegression only. "
            "On macOS, install OpenMP with `brew install libomp` to enable XGBoost."
        )

    with st.expander("How this works", expanded=False):
        st.markdown(
            """
            - Train two models: `LinearRegression` and `XGBRegressor`
            - Compare validation RMSE on an 80/20 split
            - Use the better model (lower RMSE)
            - Predict one-row test data and compute quotes:
              - `Bid = Prediction - (RMSE × risk_multiplier)`
              - `Ask = Prediction + (RMSE × risk_multiplier)`
            """
        )

    target_col = st.text_input("Target column name", value=TARGET_COLUMN_DEFAULT)
    risk_multiplier = st.slider(
        "Risk Multiplier",
        min_value=0.10,
        max_value=3.00,
        value=DEFAULT_RISK_MULTIPLIER,
        step=0.05,
    )
    st.caption(
        "Set risk_multiplier < 1.0 for aggressive/tight quotes, and > 1.0 for conservative/wider quotes."
    )

    if "trained_bundle" not in st.session_state:
        st.session_state["trained_bundle"] = None

    st.subheader("1) Upload training CSV")
    train_file = st.file_uploader("Choose train.csv", type=["csv"], key="train_uploader")

    if st.button("2) Train + Validate Models", use_container_width=True):
        try:
            train_df = load_uploaded_csv(train_file)
            validate_train_df(train_df, target_col)
            X, y = prepare_features_and_target(train_df, target_col)
            models, rmse_scores, best_model_name = train_and_evaluate_models(X, y)

            st.session_state["trained_bundle"] = {
                "models": models,
                "rmse_scores": rmse_scores,
                "best_model_name": best_model_name,
                "feature_cols": X.columns,
            }

            st.success("Training complete. Upload test.csv next.")
            st.write("Validation RMSE")
            metrics_payload = {
                "LinearRegression": round(rmse_scores["LinearRegression"], 6),
                "Best Model": best_model_name,
            }
            if "XGBRegressor" in rmse_scores:
                metrics_payload["XGBRegressor"] = round(rmse_scores["XGBRegressor"], 6)
            st.json(metrics_payload)

            if not XGBOOST_AVAILABLE and XGBOOST_IMPORT_ERROR:
                st.info(f"XGBoost import detail: {XGBOOST_IMPORT_ERROR}")

        except Exception as exc:
            st.error(f"Training failed: {exc}")

    st.subheader("3) Upload test CSV (exactly 1 row)")
    test_file = st.file_uploader("Choose test.csv", type=["csv"], key="test_uploader")

    if st.button("4) Generate Submission Quotes", use_container_width=True):
        bundle = st.session_state.get("trained_bundle")
        if bundle is None:
            st.error("Please train models first.")
            return

        try:
            test_df = load_uploaded_csv(test_file)
            X_test = prepare_test_features(test_df, bundle["feature_cols"])

            best_model_name = bundle["best_model_name"]
            best_model = bundle["models"][best_model_name]
            rmse = float(bundle["rmse_scores"][best_model_name])

            prediction = float(best_model.predict(X_test)[0])
            bid, ask, total_spread = generate_quotes(prediction, rmse, risk_multiplier)

            st.success("Quotes generated successfully.")
            st.markdown("### --- SUBMISSION QUOTES ---")
            st.write(f"Model Used: {best_model_name}")
            st.write(f"Predicted Price: {prediction:.6f}")
            st.write(f"Model Uncertainty (RMSE): {rmse:.6f}")
            st.write(f"Bid: {bid:.6f}")
            st.write(f"Ask: {ask:.6f}")
            st.write(f"Total Spread: {total_spread:.6f}")

        except Exception as exc:
            st.error(f"Quote generation failed: {exc}")


if __name__ == "__main__":
    main()
