from dataclasses import dataclass

import polars as pl


FLOAT_METRIC_ROUND_DIGITS = 2


STEP_CONFIG: dict[str, dict[str, str | pl.Expr | None]] = {
    "created_to_mql": {
        "scope_filter": None,
        "target_col": "is_mql",
        "label": "Created → MQL",
    },
    "mql_to_sql": {
        "scope_filter": pl.col("is_mql"),
        "target_col": "is_sql",
        "label": "MQL → SQL",
    },
    "mql_to_won": {
        "scope_filter": pl.col("is_mql"),
        "target_col": "is_closed_won",
        "label": "MQL → Closed Won",
    },
    "sql_to_opp": {
        "scope_filter": pl.col("is_sql"),
        "target_col": "is_opportunity",
        "label": "SQL → Opportunity",
    },
    "sql_to_won": {
        "scope_filter": pl.col("is_sql"),
        "target_col": "is_closed_won",
        "label": "SQL → Closed Won",
    },
    "opp_to_won": {
        "scope_filter": pl.col("is_opportunity"),
        "target_col": "is_closed_won",
        "label": "Opportunity → Closed Won",
    },
}


NUMERIC_FEATURES_DEFAULT = [
    "DEAL_DEMO_SCORE",
    "DEAL_AMOUNT",
    "DEAL_NUMBER_OF_EMPLOYEES",
    "DEAL_NUMERO_CEDOLINI",
    "DEAL_NUMBER_TIMES_CONTACTED",
    "COMPANY_REVENUE",
    "COMPANY_FUNDING_YEAR",
]

CATEGORICAL_FEATURES_DEFAULT = [
    "DEAL_DEALSOURCE",
    "DEAL_SOURCE_DETAIL",
    "UTM_SOURCE",
    "LEAD_TYPE",
    "DEAL_INDUSTRY",
    "CONTACT_ROLE",
    "COMPANY_STATE",
    "DEAL_HRIS_TECH_STACK",
    "DEAL_CCNL_MACRO",
]

DURATION_FEATURES_DEFAULT = [
    "creation_to_mql_days",
    "mql_to_sql_days",
    "sql_to_opp_days",
    "opp_to_won_days",
    "opp_to_lost_days",
    "creation_to_won_days",
    "creation_to_lost_days",
    "creation_to_closed_days",
]


@dataclass
class FunnelMetrics:
    volumes: dict[str, int]
    step_conversion_rates: dict[str, float | None]
    vs_created_conversion_rates: dict[str, float | None]
    stage_to_outcome_conversion_rates: dict[str, float | None]


@dataclass
class FunnelProcessMetrics:
    sql_to_demo_score_rate: float | None


def get_step_scope(df: pl.DataFrame, step: str | None) -> pl.DataFrame:
    if step is None:
        return df

    config = STEP_CONFIG[step]
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
        return round(num / den, FLOAT_METRIC_ROUND_DIGITS)

    step_rates = {
        "created_to_mql": safe_rate(volumes["mql"], volumes["created"]),
        "mql_to_sql": safe_rate(volumes["sql"], volumes["mql"]),
        "sql_to_opp": safe_rate(volumes["opportunity"], volumes["sql"]),
        "opp_to_won": safe_rate(volumes["closed_won"], volumes["opportunity"]),
    }

    stage_to_outcome_rates = {
        "mql_to_won": safe_rate(volumes["closed_won"], volumes["mql"]),
        "sql_to_won": safe_rate(volumes["closed_won"], volumes["sql"]),    }

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
                .then(
                    (pl.col("mql") / pl.col("created")).round(FLOAT_METRIC_ROUND_DIGITS)
                )
                .otherwise(None)
                .alias("creation_to_mql"),
                pl.when(pl.col("mql") > 0)
                .then((pl.col("sql") / pl.col("mql")).round(FLOAT_METRIC_ROUND_DIGITS))
                .otherwise(None)
                .alias("mql_to_sql"),
                pl.when(pl.col("sql") > 0)
                .then(
                    (pl.col("opportunity") / pl.col("sql")).round(
                        FLOAT_METRIC_ROUND_DIGITS
                    )
                )
                .otherwise(None)
                .alias("sql_to_opp"),
                pl.when(pl.col("opportunity") > 0)
                .then(
                    (pl.col("closed_won") / pl.col("opportunity")).round(
                        FLOAT_METRIC_ROUND_DIGITS
                    )
                )
                .otherwise(None)
                .alias("opp_to_won"),
                pl.when(pl.col("created") > 0)
                .then(
                    (pl.col("sql") / pl.col("created")).round(FLOAT_METRIC_ROUND_DIGITS)
                )
                .otherwise(None)
                .alias("creation_to_sql"),
                pl.when(pl.col("created") > 0)
                .then(
                    (pl.col("opportunity") / pl.col("created")).round(
                        FLOAT_METRIC_ROUND_DIGITS
                    )
                )
                .otherwise(None)
                .alias("creation_to_opportunity"),
                pl.when(pl.col("created") > 0)
                .then(
                    (pl.col("closed_won") / pl.col("created")).round(
                        FLOAT_METRIC_ROUND_DIGITS
                    )
                )
                .otherwise(None)
                .alias("creation_to_won"),
                pl.when(pl.col("created") > 0)
                .then(
                    (pl.col("closed_lost") / pl.col("created")).round(
                        FLOAT_METRIC_ROUND_DIGITS
                    )
                )
                .otherwise(None)
                .alias("creation_to_lost"),
                pl.when(pl.col("mql") > 0)
                .then(
                    (pl.col("closed_won") / pl.col("mql")).round(
                        FLOAT_METRIC_ROUND_DIGITS
                    )
                )
                .otherwise(None)
                .alias("mql_to_won"),
                pl.when(pl.col("sql") > 0)
                .then(
                    (pl.col("closed_won") / pl.col("sql")).round(
                        FLOAT_METRIC_ROUND_DIGITS
                    )
                )
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
                .round(FLOAT_METRIC_ROUND_DIGITS)
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
                pl.col(feature).mean().round(FLOAT_METRIC_ROUND_DIGITS).alias("mean"),
                pl.col(feature)
                .median()
                .round(FLOAT_METRIC_ROUND_DIGITS)
                .alias("median"),
                pl.col(feature).min().round(FLOAT_METRIC_ROUND_DIGITS).alias("min"),
                pl.col(feature).max().round(FLOAT_METRIC_ROUND_DIGITS).alias("max"),
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
                pl.col(feature).mean().round(FLOAT_METRIC_ROUND_DIGITS).alias("mean"),
                pl.col(feature)
                .median()
                .round(FLOAT_METRIC_ROUND_DIGITS)
                .alias("median"),
                pl.col(feature).min().round(FLOAT_METRIC_ROUND_DIGITS).alias("min"),
                pl.col(feature).max().round(FLOAT_METRIC_ROUND_DIGITS).alias("max"),
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
                pl.col(target_col)
                .mean()
                .round(FLOAT_METRIC_ROUND_DIGITS)
                .alias("conversion_rate"),
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
                pl.col(target_col)
                .mean()
                .round(FLOAT_METRIC_ROUND_DIGITS)
                .alias("conversion_rate"),
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
                    pl.col(col)
                    .mean()
                    .round(FLOAT_METRIC_ROUND_DIGITS)
                    .alias(f"{col}__mean"),
                    pl.col(col)
                    .median()
                    .round(FLOAT_METRIC_ROUND_DIGITS)
                    .alias(f"{col}__median"),
                    pl.col(col)
                    .min()
                    .round(FLOAT_METRIC_ROUND_DIGITS)
                    .alias(f"{col}__min"),
                    pl.col(col)
                    .max()
                    .round(FLOAT_METRIC_ROUND_DIGITS)
                    .alias(f"{col}__max"),
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
                    pl.col(col)
                    .mean()
                    .round(FLOAT_METRIC_ROUND_DIGITS)
                    .alias(f"{col}__mean"),
                    pl.col(col)
                    .median()
                    .round(FLOAT_METRIC_ROUND_DIGITS)
                    .alias(f"{col}__median"),
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
    return round(float(value), FLOAT_METRIC_ROUND_DIGITS)


def compute_process_metrics(df: pl.DataFrame) -> FunnelProcessMetrics:
    return FunnelProcessMetrics(sql_to_demo_score_rate=get_sql_to_demo_score_rate(df))


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
