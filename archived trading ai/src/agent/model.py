from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class ModelMetrics:
    rmse: float
    mae: float


def train_regressor(x: np.ndarray, y: np.ndarray, random_state: int = 42) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x, y)
    return model


def evaluate_regressor(y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return ModelMetrics(rmse=rmse, mae=mae)


def save_model(model, model_path: str | Path) -> None:
    output = Path(model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        pickle.dump(model, file)


def load_model(model_path: str | Path):
    with Path(model_path).open("rb") as file:
        model = pickle.load(file)
    return model
