from __future__ import annotations

import numpy as np


def build_pair_train_arrays(train_df, feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = train_df[feature_columns].to_numpy()
    y = train_df["target"].to_numpy()
    return x, y
