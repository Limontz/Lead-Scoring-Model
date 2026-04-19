import polars as pl
from lead_scoring.analysis.api import FunnelAnalysisResult, TemporalAnalysisResult
from lead_scoring.analysis.defaults import (
    get_categorical_features_default,
    get_duration_features_default,
    get_numeric_features_default,
    get_step_config,
    get_stage_month_cols,
    get_temporal_conversion_cols_default,
    get_temporal_duration_cols_default,
)
from lead_scoring.analysis.funnel_static import (
    compute_funnel_metrics,
    compute_process_metrics,
    compute_segment_funnel_metrics,
    compute_segment_sql_to_demo_score_rate,
    get_step_scope,
    summarize_duration_columns,
    summarize_duration_columns_by_segment,
)
from lead_scoring.analysis.funnel_temporal import (
    build_temporal_summary,
    compute_monthly_created_cohort_metrics,
    compute_monthly_duration_trends,
    compute_monthly_segment_conversion_trends,
    compute_monthly_segment_duration_trends,
    compute_monthly_segment_stage_entries,
    compute_monthly_sql_to_demo_score_rate,
    compute_monthly_stage_entries,
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
        config = get_step_config()[step]
        step_label = str(config["label"])
        target_col = str(config["target_col"])

    overall_metrics = compute_funnel_metrics(df)
    process_metrics = compute_process_metrics(df)

    duration_features = duration_features or get_duration_features_default()
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
        numeric_features=numeric_features or get_numeric_features_default(),
        categorical_features=categorical_features or get_categorical_features_default(),
        duration_features=duration_features,
    )


def analyze_temporal(
    df: pl.DataFrame,
    *,
    segment_col: str | None = None,
    conversion_cols: list[str] | None = None,
    duration_cols: list[str] | None = None,
) -> TemporalAnalysisResult:

    conversion_cols = conversion_cols or get_temporal_conversion_cols_default()
    duration_cols = duration_cols or get_temporal_duration_cols_default()

    summary_stats = build_temporal_summary(df)

    monthly_stage_entries = compute_monthly_stage_entries(
        df,
        stage_month_cols=get_stage_month_cols(),
    )
    monthly_created_cohort = compute_monthly_created_cohort_metrics(df)
    monthly_duration_trends = compute_monthly_duration_trends(
        df,
        duration_cols=duration_cols,
    )
    monthly_sql_to_demo_score = compute_monthly_sql_to_demo_score_rate(df)

    if segment_col is None:
        monthly_segment_stage_entries = None
        monthly_segment_created_cohort = None
        monthly_segment_duration_trends = None
    else:
        monthly_segment_stage_entries = compute_monthly_segment_stage_entries(
            df,
            segment_col=segment_col,
            stage_month_cols=get_stage_month_cols(),
        )
        monthly_segment_created_cohort = compute_monthly_segment_conversion_trends(
            df,
            segment_col=segment_col,
        )
        monthly_segment_duration_trends = compute_monthly_segment_duration_trends(
            df,
            segment_col=segment_col,
            duration_cols=duration_cols,
        )

    return TemporalAnalysisResult(
        full_df=df,
        temporal_df=df,
        segment_col=segment_col,
        summary_stats=summary_stats,
        monthly_stage_entries=monthly_stage_entries,
        monthly_created_cohort=monthly_created_cohort,
        monthly_duration_trends=monthly_duration_trends,
        monthly_sql_to_demo_score=monthly_sql_to_demo_score,
        monthly_segment_stage_entries=monthly_segment_stage_entries,
        monthly_segment_created_cohort=monthly_segment_created_cohort,
        monthly_segment_duration_trends=monthly_segment_duration_trends,
        conversion_cols=conversion_cols,
        duration_cols=duration_cols,
    )
