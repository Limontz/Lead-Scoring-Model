import logging
from typing import Any

import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from lead_scoring.scoring_model.config import ScoringModelConfig


logger = logging.getLogger(__name__)


def build_preprocessing_pipeline(
    categorical_features: list[str],
) -> ColumnTransformer:

    return ColumnTransformer(
        transformers=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop="if_binary",
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def build_model_pipeline(
    model: Any,
    config: ScoringModelConfig,
) -> Pipeline:

    preprocessor = build_preprocessing_pipeline(config.categorical_features)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


def stratified_train_test_split(
    df: pl.DataFrame,
    target_col: str,
    train_portion: float,
    random_state: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:

    X = df.drop(target_col)
    y = df.get_column(target_col)

    X_pd = X.to_pandas()
    y_pd = y.to_pandas()

    X_train_pd, X_test_pd, y_train_pd, y_test_pd = train_test_split(
        X_pd,
        y_pd,
        train_size=train_portion,
        stratify=y_pd,
        random_state=random_state,
    )

    X_train = pl.DataFrame(X_train_pd)
    X_test = pl.DataFrame(X_test_pd)
    y_train = pl.Series(name=target_col, values=y_train_pd)
    y_test = pl.Series(name=target_col, values=y_test_pd)

    return X_train, X_test, y_train, y_test
