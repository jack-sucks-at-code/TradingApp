from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd


@dataclass
class StockDataPair:
    stock_id: int
    train: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]


def _extract_stock_id(file_path: Path) -> int:
    match = re.search(r"stock_(\d+)_(train|test)\.csv$", file_path.name)
    if not match:
        raise ValueError(f"Invalid stock filename format: {file_path.name}")
    return int(match.group(1))


def validate_train_schema(df: pd.DataFrame, file_path: Path) -> None:
    columns = list(df.columns)
    if "target" not in columns:
        raise ValueError(
            f"Train schema mismatch in {file_path.name}. "
            f"Expected a 'target' column, got {columns}"
        )
    feature_columns = [column for column in columns if column != "target"]
    if not feature_columns:
        raise ValueError(f"Train file {file_path.name} must contain at least one feature column")
    if df.empty:
        raise ValueError(f"Train file {file_path.name} is empty")


def validate_test_schema(df: pd.DataFrame, file_path: Path, feature_columns: list[str]) -> None:
    columns = list(df.columns)
    if columns != feature_columns:
        raise ValueError(
            f"Test schema mismatch in {file_path.name}. "
            f"Expected {feature_columns}, got {columns}"
        )
    if df.empty:
        raise ValueError(f"Test file {file_path.name} is empty")


def load_stock_pair(data_dir: str | Path, stock_id: int) -> StockDataPair:
    base = Path(data_dir)
    train_path = base / f"stock_{stock_id}_train.csv"
    test_path = base / f"stock_{stock_id}_test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing train/test file pair for stock {stock_id}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    validate_train_schema(train_df, train_path)
    feature_columns = [column for column in train_df.columns if column != "target"]
    validate_test_schema(test_df, test_path, feature_columns)

    return StockDataPair(
        stock_id=stock_id,
        train=train_df,
        test=test_df,
        feature_columns=feature_columns,
    )


def discover_stock_ids(data_dir: str | Path) -> list[int]:
    base = Path(data_dir)
    train_files = sorted(base.glob("stock_*_train.csv"))
    test_files = sorted(base.glob("stock_*_test.csv"))

    if not train_files or not test_files:
        raise FileNotFoundError("No stock train/test CSV files found in data directory")

    train_ids = {_extract_stock_id(path) for path in train_files}
    test_ids = {_extract_stock_id(path) for path in test_files}

    missing_test = train_ids - test_ids
    missing_train = test_ids - train_ids

    if missing_test:
        raise ValueError(f"Missing test files for stock IDs: {sorted(missing_test)}")
    if missing_train:
        raise ValueError(f"Missing train files for stock IDs: {sorted(missing_train)}")

    return sorted(train_ids)


def load_all_stock_pairs(data_dir: str | Path) -> list[StockDataPair]:
    stock_ids = discover_stock_ids(data_dir)
    return [load_stock_pair(data_dir, stock_id) for stock_id in stock_ids]
