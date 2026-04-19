from dataclasses import dataclass

import polars as pl

from lead_scoring.analysis.defaults import (
    get_stage_month_cols,
)


@dataclass
class TemporalSummary:
    n_months: int
    first_month: str | None
    last_month: str | None


def compute_monthly_stage_entries(
    df: pl.DataFrame,
    stage_month_cols: dict[str, str] | None = None,
) -> pl.DataFrame:
    if stage_month_cols is None:
        stage_month_cols = get_stage_month_cols()
    frames: list[pl.DataFrame] = []

    for stage_name, month_col in stage_month_cols.items():
        if month_col not in df.columns:
            continue

        stage_df = (
            df.filter(pl.col(month_col).is_not_null())
            .group_by(month_col)
            .agg(pl.len().alias("n_deals"))
            .rename({month_col: "month"})
            .with_columns(pl.lit(stage_name).alias("stage"))
            .select(["month", "stage", "n_deals"])
        )
        frames.append(stage_df)

    if not frames:
        return pl.DataFrame({"month": [], "stage": [], "n_deals": []})

    return pl.concat(frames).sort(["month", "stage"])


def compute_monthly_created_cohort_metrics(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return pl.DataFrame(
            {
                "created_month": [],
                "created": [],
                "mql": [],
                "sql": [],
                "opportunity": [],
                "closed_won": [],
                "closed_lost": [],
                "cr_created_to_mql": [],
                "cr_mql_to_sql": [],
                "cr_sql_to_opp": [],
                "cr_opp_to_won": [],
                "cr_created_to_won": [],
                "cr_created_to_lost": [],
            }
        )

    return (
        df.filter(pl.col("created_month").is_not_null())
        .group_by("created_month")
        .agg(
            [
                pl.len().alias("created"),
                pl.col("is_mql").sum().alias("mql"),
                pl.col("is_sql").sum().alias("sql"),
                pl.col("is_opportunity").sum().alias("opportunity"),
                pl.col("is_closed_won").sum().alias("closed_won"),
                pl.col("is_closed_lost").sum().alias("closed_lost"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("created") > 0)
                .then((pl.col("mql") / pl.col("created")).round(2))
                .otherwise(None)
                .alias("cr_created_to_mql"),
                pl.when(pl.col("mql") > 0)
                .then((pl.col("sql") / pl.col("mql")).round(2))
                .otherwise(None)
                .alias("cr_mql_to_sql"),
                pl.when(pl.col("sql") > 0)
                .then((pl.col("opportunity") / pl.col("sql")).round(2))
                .otherwise(None)
                .alias("cr_sql_to_opp"),
                pl.when(pl.col("opportunity") > 0)
                .then((pl.col("closed_won") / pl.col("opportunity")).round(2))
                .otherwise(None)
                .alias("cr_opp_to_won"),
                pl.when(pl.col("created") > 0)
                .then((pl.col("closed_won") / pl.col("created")).round(2))
                .otherwise(None)
                .alias("cr_created_to_won"),
                pl.when(pl.col("created") > 0)
                .then((pl.col("closed_lost") / pl.col("created")).round(2))
                .otherwise(None)
                .alias("cr_created_to_lost"),
            ]
        )
        .sort("created_month")
    )


def compute_monthly_duration_trends(
    df: pl.DataFrame,
    duration_cols: list[str],
) -> pl.DataFrame:
    agg_exprs: list[pl.Expr] = []

    for col in duration_cols:
        if col in df.columns:
            agg_exprs.append(pl.col(col).mean().round(2).alias(col))

    if not agg_exprs:
        return pl.DataFrame({"created_month": []})

    return (
        df.filter(pl.col("created_month").is_not_null())
        .group_by("created_month")
        .agg(agg_exprs)
        .sort("created_month")
    )


def compute_monthly_sql_to_demo_score_rate(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.filter(pl.col("created_month").is_not_null())
        .filter(pl.col("is_sql"))
        .group_by("created_month")
        .agg(
            [
                pl.len().alias("sql_count"),
                pl.col("DEAL_DEMO_SCORE")
                .is_not_null()
                .mean()
                .round(2)
                .alias("sql_to_demo_score_rate"),
            ]
        )
        .sort("created_month")
    )


def compute_monthly_segment_conversion_trends(
    df: pl.DataFrame,
    segment_col: str,
) -> pl.DataFrame:
    return (
        df.filter(pl.col("created_month").is_not_null())
        .group_by(["created_month", segment_col])
        .agg(
            [
                pl.len().alias("created"),
                pl.col("is_mql").sum().alias("mql"),
                pl.col("is_sql").sum().alias("sql"),
                pl.col("is_opportunity").sum().alias("opportunity"),
                pl.col("is_closed_won").sum().alias("closed_won"),
                pl.col("is_closed_lost").sum().alias("closed_lost"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("created") > 0)
                .then((pl.col("closed_won") / pl.col("created")).round(2))
                .otherwise(None)
                .alias("cr_created_to_won"),
                pl.when(pl.col("created") > 0)
                .then((pl.col("mql") / pl.col("created")).round(2))
                .otherwise(None)
                .alias("cr_created_to_mql"),
                pl.when(pl.col("mql") > 0)
                .then((pl.col("sql") / pl.col("mql")).round(2))
                .otherwise(None)
                .alias("cr_mql_to_sql"),
                pl.when(pl.col("sql") > 0)
                .then((pl.col("opportunity") / pl.col("sql")).round(2))
                .otherwise(None)
                .alias("cr_sql_to_opp"),
                pl.when(pl.col("opportunity") > 0)
                .then((pl.col("closed_won") / pl.col("opportunity")).round(2))
                .otherwise(None)
                .alias("cr_opp_to_won"),
            ]
        )
        .sort(["created_month", segment_col])
    )


def compute_monthly_segment_stage_entries(
    df: pl.DataFrame,
    segment_col: str,
    stage_month_cols: dict[str, str] | None = None,
) -> pl.DataFrame:
    if stage_month_cols is None:
        stage_month_cols = get_stage_month_cols()
    frames: list[pl.DataFrame] = []

    for stage_name, month_col in stage_month_cols.items():
        if month_col not in df.columns:
            continue

        stage_df = (
            df.filter(pl.col(month_col).is_not_null())
            .group_by([month_col, segment_col])
            .agg(pl.len().alias("n_deals"))
            .rename({month_col: "month"})
            .with_columns(pl.lit(stage_name).alias("stage"))
            .select(["month", segment_col, "stage", "n_deals"])
        )
        frames.append(stage_df)

    if not frames:
        return pl.DataFrame({"month": [], segment_col: [], "stage": [], "n_deals": []})

    return pl.concat(frames).sort(["month", segment_col, "stage"])


def compute_monthly_segment_duration_trends(
    df: pl.DataFrame,
    segment_col: str,
    duration_cols: list[str],
) -> pl.DataFrame:
    agg_exprs: list[pl.Expr] = []

    for col in duration_cols:
        if col in df.columns:
            agg_exprs.append(pl.col(col).mean().round(2).alias(col))

    if not agg_exprs:
        return pl.DataFrame({"created_month": [], segment_col: []})

    return (
        df.filter(pl.col("created_month").is_not_null())
        .group_by(["created_month", segment_col])
        .agg(agg_exprs)
        .sort(["created_month", segment_col])
    )


def build_temporal_summary(df: pl.DataFrame) -> TemporalSummary:
    if df.is_empty():
        return TemporalSummary(n_months=0, first_month=None, last_month=None)

    month_stats = df.select(
        [
            pl.col("created_month").n_unique().alias("n_months"),
            pl.col("created_month").min().alias("first_month"),
            pl.col("created_month").max().alias("last_month"),
        ]
    ).to_dicts()[0]

    return TemporalSummary(
        n_months=int(month_stats["n_months"] or 0),
        first_month=None
        if month_stats["first_month"] is None
        else str(month_stats["first_month"]),
        last_month=None
        if month_stats["last_month"] is None
        else str(month_stats["last_month"]),
    )
