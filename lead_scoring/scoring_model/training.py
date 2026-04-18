import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from lead_scoring.registry import ModelName
from lead_scoring.scoring_model.config import ScoringModelConfig
from lead_scoring.scoring_model.pipeline import build_model_pipeline


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


def prepare_features_for_model(
    X_pd: pd.DataFrame,
    model_name: ModelName,
    categorical_features: list[str] | None = None,
) -> pd.DataFrame:
    if model_name != ModelName.XGBOOST:
        return X_pd

    prepared = X_pd.copy()

    for col in categorical_features or []:
        if col in prepared.columns:
            prepared[col] = prepared[col].astype("category")

    return prepared


def train_model(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    config: ScoringModelConfig,
) -> Pipeline:
    pipeline = build_model_pipeline(config)
    X_train_pd = prepare_features_for_model(
        X_train.to_pandas(),
        model_name=config.model_name,
        categorical_features=config.categorical_features,
    )
    pipeline.fit(X_train_pd, y_train.to_numpy())
    return pipeline
