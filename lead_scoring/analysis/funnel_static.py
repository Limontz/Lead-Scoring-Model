from dataclasses import dataclass, field

import polars as pl

from lead_scoring.analysis.defaults import (
    get_step_config,
)


@dataclass
class FunnelMetrics:
    volumes: dict[str, int]
    step_conversion_rates: dict[str, float | None]
    vs_created_conversion_rates: dict[str, float | None]
    stage_to_outcome_conversion_rates: dict[str, float | None]
    avg_deal_amount_by_stage: dict[str, float | None] = field(default_factory=dict)


@dataclass
class FunnelProcessMetrics:
    sql_to_demo_score_rate: float | None


def get_step_scope(df: pl.DataFrame, step: str | None) -> pl.DataFrame:
    if step is None:
        return df

    config = get_step_config()[step]
    if config["scope_filter"] is None:
        return df
    return df.filter(config["scope_filter"])


def compute_funnel_metrics(df: pl.DataFrame) -> FunnelMetrics:
    volumes = {
        "created": df.height,
        "mql": int(df["is_mql"].sum()),
        "sql": int(df["is_sql"].sum()),
        "opportunity": int(df["is_opportunity"].sum()),
        "closed_won": int(df["is_closed_won"].sum()),
        "closed_lost": int(df["is_closed_lost"].sum()),
    }

    def safe_rate(num: int, den: int) -> float | None:
        if den == 0:
            return None
        return round(num / den, 2)

    step_rates = {
        "created_to_mql": safe_rate(volumes["mql"], volumes["created"]),
        "mql_to_sql": safe_rate(volumes["sql"], volumes["mql"]),
        "sql_to_opp": safe_rate(volumes["opportunity"], volumes["sql"]),
        "opp_to_won": safe_rate(volumes["closed_won"], volumes["opportunity"]),
    }

    stage_to_outcome_rates = {
        "mql_to_won": safe_rate(volumes["closed_won"], volumes["mql"]),
        "sql_to_won": safe_rate(volumes["closed_won"], volumes["sql"]),
    }

    vs_created_rates = {
        "creation_to_mql": safe_rate(volumes["mql"], volumes["created"]),
        "creation_to_sql": safe_rate(volumes["sql"], volumes["created"]),
        "creation_to_opportunity": safe_rate(
            volumes["opportunity"], volumes["created"]
        ),
        "creation_to_won": safe_rate(volumes["closed_won"], volumes["created"]),
        "creation_to_lost": safe_rate(volumes["closed_lost"], volumes["created"]),
    }

    return FunnelMetrics(
        volumes=volumes,
        step_conversion_rates=step_rates,
        vs_created_conversion_rates=vs_created_rates,
        stage_to_outcome_conversion_rates=stage_to_outcome_rates,
        avg_deal_amount_by_stage=compute_avg_deal_amount_by_stage(df),
    )


def compute_segment_funnel_metrics(
    df: pl.DataFrame,
    segment_col: str,
) -> pl.DataFrame:
    return (
        df.group_by(segment_col)
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
                .alias("creation_to_mql"),
                pl.when(pl.col("mql") > 0)
                .then((pl.col("sql") / pl.col("mql")).round(2))
                .otherwise(None)
                .alias("mql_to_sql"),
                pl.when(pl.col("sql") > 0)
                .then((pl.col("opportunity") / pl.col("sql")).round(2))
                .otherwise(None)
                .alias("sql_to_opp"),
                pl.when(pl.col("opportunity") > 0)
                .then((pl.col("closed_won") / pl.col("opportunity")).round(2))
                .otherwise(None)
                .alias("opp_to_won"),
                pl.when(pl.col("created") > 0)
                .then((pl.col("sql") / pl.col("created")).round(2))
                .otherwise(None)
                .alias("creation_to_sql"),
                pl.when(pl.col("created") > 0)
                .then((pl.col("opportunity") / pl.col("created")).round(2))
                .otherwise(None)
                .alias("creation_to_opportunity"),
                pl.when(pl.col("created") > 0)
                .then((pl.col("closed_won") / pl.col("created")).round(2))
                .otherwise(None)
                .alias("creation_to_won"),
                pl.when(pl.col("created") > 0)
                .then((pl.col("closed_lost") / pl.col("created")).round(2))
                .otherwise(None)
                .alias("creation_to_lost"),
                pl.when(pl.col("mql") > 0)
                .then((pl.col("closed_won") / pl.col("mql")).round(2))
                .otherwise(None)
                .alias("mql_to_won"),
                pl.when(pl.col("sql") > 0)
                .then((pl.col("closed_won") / pl.col("sql")).round(2))
                .otherwise(None)
                .alias("sql_to_won"),
            ]
        )
        .sort("created", descending=True)
    )


def compute_segment_sql_to_demo_score_rate(
    df: pl.DataFrame,
    segment_col: str,
) -> pl.DataFrame:
    return (
        df.filter(pl.col("is_sql"))
        .group_by(segment_col)
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
        .sort("sql_count", descending=True)
    )


def summarize_numeric_by_target(
    df: pl.DataFrame,
    feature: str,
    target_col: str,
) -> pl.DataFrame:
    return (
        df.filter(pl.col(feature).is_not_null())
        .group_by(target_col)
        .agg(
            [
                pl.len().alias("n"),
                pl.col(feature).mean().round(2).alias("mean"),
                pl.col(feature).median().round(2).alias("median"),
                pl.col(feature).min().round(2).alias("min"),
                pl.col(feature).max().round(2).alias("max"),
            ]
        )
        .sort(target_col)
    )


def summarize_numeric_by_target_and_segment(
    df: pl.DataFrame,
    feature: str,
    target_col: str,
    segment_col: str,
) -> pl.DataFrame:
    return (
        df.filter(pl.col(feature).is_not_null())
        .group_by([target_col, segment_col])
        .agg(
            [
                pl.len().alias("n"),
                pl.col(feature).mean().round(2).alias("mean"),
                pl.col(feature).median().round(2).alias("median"),
                pl.col(feature).min().round(2).alias("min"),
                pl.col(feature).max().round(2).alias("max"),
            ]
        )
        .sort([target_col, segment_col])
    )


def summarize_categorical_conversion(
    df: pl.DataFrame,
    feature: str,
    target_col: str,
    min_count: int = 0,
) -> pl.DataFrame:
    return (
        df.filter(pl.col(feature).is_not_null())
        .group_by(feature)
        .agg(
            [
                pl.len().alias("n"),
                pl.col(target_col).mean().round(2).alias("conversion_rate"),
            ]
        )
        .filter(pl.col("n") >= min_count)
        .sort("conversion_rate", descending=True)
    )


def summarize_categorical_conversion_by_segment(
    df: pl.DataFrame,
    feature: str,
    target_col: str,
    segment_col: str,
    min_count: int = 0,
) -> pl.DataFrame:
    return (
        df.filter(pl.col(feature).is_not_null())
        .group_by([feature, segment_col])
        .agg(
            [
                pl.len().alias("n"),
                pl.col(target_col).mean().round(2).alias("conversion_rate"),
            ]
        )
        .filter(pl.col("n") >= min_count)
        .sort(["conversion_rate", "n"], descending=[True, True])
    )


def summarize_duration_columns(
    df: pl.DataFrame,
    duration_cols: list[str],
) -> pl.DataFrame:
    agg_exprs: list[pl.Expr] = []
    for col in duration_cols:
        if col in df.columns:
            agg_exprs.extend(
                [
                    pl.col(col).count().alias(f"{col}__n"),
                    pl.col(col).mean().round(2).alias(f"{col}__mean"),
                    pl.col(col).median().round(2).alias(f"{col}__median"),
                    pl.col(col).min().round(2).alias(f"{col}__min"),
                    pl.col(col).max().round(2).alias(f"{col}__max"),
                ]
            )

    return df.select(agg_exprs)


def summarize_duration_columns_by_segment(
    df: pl.DataFrame,
    segment_col: str,
    duration_cols: list[str],
) -> pl.DataFrame:
    agg_exprs: list[pl.Expr] = []
    for col in duration_cols:
        if col in df.columns:
            agg_exprs.extend(
                [
                    pl.col(col).count().alias(f"{col}__n"),
                    pl.col(col).mean().round(2).alias(f"{col}__mean"),
                    pl.col(col).median().round(2).alias(f"{col}__median"),
                ]
            )

    return df.group_by(segment_col).agg(agg_exprs).sort(segment_col)


def summarize_closed_lost_reasons(
    df: pl.DataFrame,
    reason_col: str = "DEAL_CLOSED_LOST_REASON",
    top_n: int = 99999,
) -> pl.DataFrame:
    return (
        df.filter(pl.col("is_closed_lost"))
        .filter(pl.col(reason_col).is_not_null())
        .group_by(reason_col)
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(top_n)
    )


def summarize_closed_lost_reasons_by_segment(
    df: pl.DataFrame,
    segment_col: str,
    reason_col: str = "DEAL_CLOSED_LOST_REASON",
    top_n: int = 99999,
) -> pl.DataFrame:
    return (
        df.filter(pl.col("is_closed_lost"))
        .filter(pl.col(reason_col).is_not_null())
        .group_by([reason_col, segment_col])
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(top_n)
    )


def get_sql_to_demo_score_rate(df: pl.DataFrame) -> float | None:
    sql_df = df.filter(pl.col("is_sql"))
    value = sql_df.select(pl.col("DEAL_DEMO_SCORE").is_not_null().mean()).item()
    return round(float(value), 2)


def compute_process_metrics(df: pl.DataFrame) -> FunnelProcessMetrics:
    return FunnelProcessMetrics(sql_to_demo_score_rate=get_sql_to_demo_score_rate(df))


def compute_avg_deal_amount_by_stage(df: pl.DataFrame) -> dict[str, float | None]:
    created_avg = df.select(pl.col("DEAL_AMOUNT").mean()).item()
    mql_avg = df.filter(pl.col("is_mql")).select(pl.col("DEAL_AMOUNT").mean()).item()
    sql_avg = df.filter(pl.col("is_sql")).select(pl.col("DEAL_AMOUNT").mean()).item()
    opportunity_avg = (
        df.filter(pl.col("is_opportunity")).select(pl.col("DEAL_AMOUNT").mean()).item()
    )
    closed_won_avg = (
        df.filter(pl.col("is_closed_won")).select(pl.col("DEAL_AMOUNT").mean()).item()
    )
    closed_lost_avg = (
        df.filter(pl.col("is_closed_lost")).select(pl.col("DEAL_AMOUNT").mean()).item()
    )

    return {
        "created": round(float(created_avg), 2),
        "mql": round(float(mql_avg), 2),
        "sql": round(float(sql_avg), 2),
        "opportunity": round(float(opportunity_avg), 2),
        "closed_won": round(float(closed_won_avg), 2),
        "closed_lost": round(float(closed_lost_avg), 2),
    }


def compute_segment_stage_volumes_long(
    segment_metrics_df: pl.DataFrame,
    segment_col: str,
) -> pl.DataFrame:
    return segment_metrics_df.select(
        [
            segment_col,
            "created",
            "mql",
            "sql",
            "opportunity",
            "closed_won",
            "closed_lost",
        ]
    ).unpivot(
        on=["created", "mql", "sql", "opportunity", "closed_won", "closed_lost"],
        index=segment_col,
        variable_name="stage",
        value_name="n_deals",
    )


def compute_segment_conversion_long(
    segment_metrics_df: pl.DataFrame,
    segment_col: str,
    conversion_cols: list[str],
) -> pl.DataFrame:
    return segment_metrics_df.select([segment_col] + conversion_cols).unpivot(
        on=conversion_cols,
        index=segment_col,
        variable_name="transition",
        value_name="conversion_rate",
    )
