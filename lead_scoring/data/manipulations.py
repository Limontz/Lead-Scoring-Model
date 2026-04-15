import polars as pl
import pandera.polars as pa
from pandera.typing.polars import DataFrame

from lead_scoring.data.schema import EnrichedDealsSchema, RawDealsSchemaWithDatetime

FUNNEL_STAGE_DATE_COLS = {
    "created": "DEAL_CREATEDATE",
    "mql": "DEAL_MQL_DATETIME",
    "sql": "DEAL_SQL_DATETIME",
    "opportunity": "DEAL_OPPORTUNITY_DATETIME",
    "closed_won": "DEAL_CLOSED_WON_DATE",
    "closed_lost": "DEAL_DATETIME_ENTERED_CLOSEDLOST",
}


def _add_stage_flags(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.col("DEAL_MQL_DATETIME").is_not_null().alias("is_mql"),
            pl.col("DEAL_SQL_DATETIME").is_not_null().alias("is_sql"),
            pl.col("DEAL_OPPORTUNITY_DATETIME").is_not_null().alias("is_opportunity"),
            pl.col("DEAL_CLOSED_WON_DATE").is_not_null().alias("is_closed_won"),
            pl.col("DEAL_DATETIME_ENTERED_CLOSEDLOST").is_not_null().alias("is_closed_lost"),
        ]
    )


def _add_final_state_columns(df: pl.DataFrame) -> pl.DataFrame:
    closed_at_expr = pl.coalesce(
        [
            pl.col("DEAL_CLOSED_WON_DATE"),
            pl.col("DEAL_DATETIME_ENTERED_CLOSEDLOST"),
        ]
    )

    return df.with_columns(
        [
            (pl.col("is_closed_won") | pl.col("is_closed_lost")).alias("has_final_state"),
            (~pl.col("is_closed_won") & ~pl.col("is_closed_lost")).alias("is_open"),
            (
                ~pl.col("is_mql")
                & ~pl.col("is_sql")
                & ~pl.col("is_opportunity")
                & ~pl.col("is_closed_won")
                & ~pl.col("is_closed_lost")
            ).alias("is_inactive"),
            pl.when(pl.col("is_closed_won"))
            .then(pl.lit("won"))
            .when(pl.col("is_closed_lost"))
            .then(pl.lit("lost"))
            .otherwise(pl.lit("open"))
            .alias("final_status"),
            closed_at_expr.alias("closed_at"),
        ]
    )


def _add_month_columns(
    df: pl.DataFrame,
    stage_date_cols: dict[str, str] = FUNNEL_STAGE_DATE_COLS,
) -> pl.DataFrame:
    exprs = [
        pl.col(date_col).dt.truncate("1mo").alias(f"{stage_name}_month")
        for stage_name, date_col in stage_date_cols.items()
    ]
    return df.with_columns(exprs)


def _add_duration_columns(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            (pl.col("DEAL_MQL_DATETIME") - pl.col("DEAL_CREATEDATE"))
            .dt.total_days()
            .alias("creation_to_mql_days"),

            (pl.col("DEAL_SQL_DATETIME") - pl.col("DEAL_MQL_DATETIME"))
            .dt.total_days()
            .alias("mql_to_sql_days"),

            (pl.col("DEAL_OPPORTUNITY_DATETIME") - pl.col("DEAL_SQL_DATETIME"))
            .dt.total_days()
            .alias("sql_to_opp_days"),

            (pl.col("DEAL_CLOSED_WON_DATE") - pl.col("DEAL_OPPORTUNITY_DATETIME"))
            .dt.total_days()
            .alias("opp_to_won_days"),

            (pl.col("DEAL_DATETIME_ENTERED_CLOSEDLOST") - pl.col("DEAL_OPPORTUNITY_DATETIME"))
            .dt.total_days()
            .alias("opp_to_lost_days"),

            (pl.col("DEAL_SQL_DATETIME") - pl.col("DEAL_CREATEDATE"))
            .dt.total_days()
            .alias("creation_to_sql_days"),

            (pl.col("DEAL_OPPORTUNITY_DATETIME") - pl.col("DEAL_CREATEDATE"))
            .dt.total_days()
            .alias("creation_to_opp_days"),

            (pl.col("DEAL_CLOSED_WON_DATE") - pl.col("DEAL_CREATEDATE"))
            .dt.total_days()
            .alias("creation_to_won_days"),

            (pl.col("DEAL_DATETIME_ENTERED_CLOSEDLOST") - pl.col("DEAL_CREATEDATE"))
            .dt.total_days()
            .alias("creation_to_lost_days"),

            (pl.col("closed_at") - pl.col("DEAL_CREATEDATE"))
            .dt.total_days()
            .alias("creation_to_closed_days"),
        ]
    )


@pa.check_types
def prepare_data_for_analysis(
    df: DataFrame[RawDealsSchemaWithDatetime],
) -> DataFrame[EnrichedDealsSchema]:
    enriched_df = (
        df.pipe(_add_stage_flags)
        .pipe(_add_final_state_columns)
        .pipe(_add_month_columns)
        .pipe(_add_duration_columns)
    )
    return DataFrame[EnrichedDealsSchema](enriched_df)