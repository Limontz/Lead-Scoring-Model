from dataclasses import dataclass

import polars as pl
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class ClassificationMetrics:
    roc_auc: float
    average_precision: float
    accuracy: float


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
