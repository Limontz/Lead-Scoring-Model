import polars as pl
from lead_scoring.analysis.api import FunnelAnalysisResult
from lead_scoring.analysis.funnel_static import (
    CATEGORICAL_FEATURES_DEFAULT,
    DURATION_FEATURES_DEFAULT,
    NUMERIC_FEATURES_DEFAULT,
    STEP_CONFIG,
    compute_funnel_metrics,
    compute_process_metrics,
    compute_segment_funnel_metrics,
    compute_segment_sql_to_demo_score_rate,
    get_step_scope,
    summarize_duration_columns,
    summarize_duration_columns_by_segment,
)


def analyze_funnel(
    df: pl.DataFrame,
    *,
    segment_col: str | None = None,
    step: str | None = None,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    duration_features: list[str] | None = None,
) -> FunnelAnalysisResult:
    scoped_df = get_step_scope(df, step=step)

    if step is None:
        step_label = "Full funnel"
        target_col = None
    else:
        config = STEP_CONFIG[step]
        step_label = str(config["label"])
        target_col = str(config["target_col"])

    overall_metrics = compute_funnel_metrics(df)
    process_metrics = compute_process_metrics(df)

    duration_features = duration_features or DURATION_FEATURES_DEFAULT
    duration_summary = summarize_duration_columns(df, duration_features)

    if segment_col is None:
        segment_metrics = None
        segment_sql_to_demo_score = None
        segment_duration_summary = None
    else:
        segment_metrics = compute_segment_funnel_metrics(df, segment_col)
        segment_sql_to_demo_score = compute_segment_sql_to_demo_score_rate(
            df, segment_col
        )
        segment_duration_summary = summarize_duration_columns_by_segment(
            df,
            segment_col,
            duration_features,
        )

    return FunnelAnalysisResult(
        full_df=df,
        scoped_df=scoped_df,
        step=step,
        step_label=step_label,
        target_col=target_col,
        segment_col=segment_col,
        overall_metrics=overall_metrics,
        process_metrics=process_metrics,
        segment_metrics=segment_metrics,
        segment_sql_to_demo_score=segment_sql_to_demo_score,
        duration_summary=duration_summary,
        segment_duration_summary=segment_duration_summary,
        numeric_features=numeric_features or NUMERIC_FEATURES_DEFAULT,
        categorical_features=categorical_features or CATEGORICAL_FEATURES_DEFAULT,
        duration_features=duration_features,
    )
