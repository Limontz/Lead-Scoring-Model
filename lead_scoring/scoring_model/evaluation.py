from dataclasses import dataclass

import polars as pl
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from lead_scoring.scoring_model.config import ScoringModelConfig
from lead_scoring.scoring_model.training import prepare_features_for_model


@dataclass(frozen=True)
class ClassificationMetrics:
    roc_auc: float
    average_precision: float
    accuracy: float


def evaluate_binary_classifier(
    pipeline: Pipeline,
    X_test: pl.DataFrame,
    y_test: pl.Series,
    config: ScoringModelConfig,
) -> ClassificationMetrics:
    X_test_pd = prepare_features_for_model(
        X_test.to_pandas(),
        model_name=config.model_name,
        categorical_features=config.categorical_features,
    )
    y_test_np = y_test.to_numpy()

    y_pred = pipeline.predict(X_test_pd)
    y_score = pipeline.predict_proba(X_test_pd)[:, 1]

    return ClassificationMetrics(
        roc_auc=roc_auc_score(y_test_np, y_score),
        average_precision=average_precision_score(y_test_np, y_score),
        accuracy=accuracy_score(y_test_np, y_pred),
    )


def get_linear_model_weights(pipeline: Pipeline) -> pl.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    if not hasattr(model, "coef_"):
        raise ValueError("Model does not expose coefficients via coef_.")

    if not hasattr(preprocessor, "get_feature_names_out"):
        raise ValueError(
            "Linear model weights are only supported with named preprocessor output."
        )

    coefficients = model.coef_
    if coefficients.ndim == 1:
        weights = coefficients
    elif coefficients.shape[0] == 1:
        weights = coefficients[0]
    else:
        raise ValueError(
            "Only binary linear models are supported for weight extraction."
        )

    feature_names = preprocessor.get_feature_names_out()

    if len(feature_names) != len(weights):
        raise ValueError("Feature names and coefficient lengths do not match.")

    return (
        pl.DataFrame({"feature": feature_names, "weight": weights})
        .with_columns(pl.col("weight").abs().alias("abs_weight"))
        .sort("abs_weight", descending=True)
    )
