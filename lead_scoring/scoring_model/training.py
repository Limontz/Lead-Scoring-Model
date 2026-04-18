from dataclasses import dataclass
from typing import Any

import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from lead_scoring.scoring_model.config import ScoringModelConfig
from lead_scoring.scoring_model.pipeline import (
    build_model_pipeline,
    stratified_train_test_split,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    roc_auc: float
    average_precision: float
    accuracy: float


def build_logistic_regression_model(
    config: ScoringModelConfig,
) -> LogisticRegression:
    return LogisticRegression(
        random_state=config.random_state,
        max_iter=1000,
    )


def evaluate_binary_classifier(
    pipeline: Pipeline,
    X_test: pl.DataFrame,
    y_test: pl.Series,
) -> ClassificationMetrics:
    X_test_pd = X_test.to_pandas()
    y_test_np = y_test.to_numpy()

    y_pred = pipeline.predict(X_test_pd)
    y_score = pipeline.predict_proba(X_test_pd)[:, 1]

    return ClassificationMetrics(
        roc_auc=roc_auc_score(y_test_np, y_score),
        average_precision=average_precision_score(y_test_np, y_score),
        accuracy=accuracy_score(y_test_np, y_pred),
    )


def fit_and_evaluate_model(
    df: pl.DataFrame,
    model: Any,
    config: ScoringModelConfig,
) -> tuple[Pipeline, ClassificationMetrics]:
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        df=df,
        target_col=config.target,
        train_portion=config.training_set_portion,
        random_state=config.random_state,
    )

    pipeline = build_model_pipeline(model, config)
    pipeline.fit(X_train.to_pandas(), y_train.to_numpy())
    metrics = evaluate_binary_classifier(pipeline, X_test, y_test)

    return pipeline, metrics


def train_logistic_regression(
    df: pl.DataFrame,
    config: ScoringModelConfig,
) -> tuple[Pipeline, ClassificationMetrics]:
    model = build_logistic_regression_model(config)
    return fit_and_evaluate_model(df=df, model=model, config=config)
