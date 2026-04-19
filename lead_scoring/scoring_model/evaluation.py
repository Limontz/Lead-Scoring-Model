from dataclasses import dataclass
from math import ceil

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
    missed_won_leads_count: int
    missed_won_leads_rate: float
    baseline_conversion_rate: float
    top_fraction_conversion_rate: float
    lift_at_top_fraction: float
    ranking_by_bucket: pl.DataFrame


def evaluate_missed_won_leads(
    y_true: pl.Series,
    y_pred: pl.Series,
) -> tuple[int, float]:
    eval_df = pl.DataFrame({"target": y_true, "pred": y_pred})
    missed_won_count = eval_df.filter(
        (pl.col("target") == 1) & (pl.col("pred") == 0)
    ).height
    total_won = eval_df.filter(pl.col("target") == 1).height

    return missed_won_count, missed_won_count / total_won


def create_score_buckets(
    y_true: pl.Series,
    y_score: pl.Series,
    n_buckets: int = 5,
) -> pl.DataFrame:
    scored_df = (
        pl.DataFrame({"target": y_true, "score": y_score})
        .sort("score", descending=True)
        .with_row_index("row_idx")
        .with_columns(
            (
                ((pl.col("row_idx") * n_buckets) / len(y_true))
                .floor()
                .clip(upper_bound=n_buckets - 1)
                .cast(pl.Int64)
                + 1
            ).alias("score_bucket")
        )
    )

    return scored_df


def evaluate_ranking_quality(
    y_true: pl.Series,
    y_score: pl.Series,
    *,
    top_fraction: float = 0.2,
    n_buckets: int = 5,
) -> tuple[float, float, float, pl.DataFrame]:
    if not (0 < top_fraction <= 1):
        raise ValueError("top_fraction must be in the (0, 1] interval.")
    if n_buckets < 2:
        raise ValueError("n_buckets must be >= 2.")

    scored_df = create_score_buckets(y_true, y_score, n_buckets)
    baseline_mean = scored_df.get_column("target").mean()
    baseline_rate = float(baseline_mean)  # pyright: ignore[reportArgumentType]
    top_n = max(1, ceil(len(y_true) * top_fraction))
    top_mean = scored_df.head(top_n).get_column("target").mean()
    top_fraction_rate = float(top_mean)  # pyright: ignore[reportArgumentType]

    lift_at_top_fraction = top_fraction_rate / baseline_rate

    ranking_by_bucket = (
        scored_df.group_by("score_bucket")
        .agg(
            pl.len().alias("lead_count"),
            pl.col("target").mean().alias("conversion_rate"),
            pl.col("score").mean().alias("avg_model_score"),
        )
        .sort("score_bucket")
    )

    return (
        baseline_rate,
        top_fraction_rate,
        lift_at_top_fraction,
        ranking_by_bucket,
    )


def evaluate_binary_classifier(
    pipeline: Pipeline,
    X_test: pl.DataFrame,
    y_test: pl.Series,
    config: ScoringModelConfig,
    *,
    top_fraction: float = 0.2,
    n_buckets: int = 5,
) -> ClassificationMetrics:
    X_test_pd = prepare_features_for_model(
        X_test.to_pandas(),
        model_name=config.model_name,
        categorical_features=config.categorical_features,
    )
    y_test_np = y_test.to_numpy()

    y_pred = pipeline.predict(X_test_pd)
    y_score = pipeline.predict_proba(X_test_pd)[:, 1]
    missed_won_count, missed_won_rate = evaluate_missed_won_leads(
        y_true=y_test,
        y_pred=pl.Series(name="prediction", values=y_pred),
    )
    (
        baseline_rate,
        top_fraction_rate,
        lift_at_top_fraction,
        ranking_by_bucket,
    ) = evaluate_ranking_quality(
        y_true=y_test,
        y_score=pl.Series(name="score", values=y_score),
        top_fraction=top_fraction,
        n_buckets=n_buckets,
    )

    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_test_np, y_score)),
        average_precision=float(average_precision_score(y_test_np, y_score)),
        accuracy=float(accuracy_score(y_test_np, y_pred)),
        missed_won_leads_count=missed_won_count,
        missed_won_leads_rate=missed_won_rate,
        baseline_conversion_rate=baseline_rate,
        top_fraction_conversion_rate=top_fraction_rate,
        lift_at_top_fraction=lift_at_top_fraction,
        ranking_by_bucket=ranking_by_bucket,
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
