from datetime import datetime

import polars as pl


def cast_datetime_columns(
    df: pl.DataFrame,
) -> pl.DataFrame:
    datetime_columns = [
        "DEAL_CREATEDATE",
        "DEAL_MQL_DATETIME",
        "DEAL_SQL_DATETIME",
        "DEAL_OPPORTUNITY_DATETIME",
        "DEAL_CLOSED_WON_DATE",
        "DEAL_DATETIME_ENTERED_CLOSEDLOST",
    ]
    datetime_format = "%Y-%m-%d %H:%M:%S"
    return df.with_columns(
        [
            pl.col(col).str.to_datetime(format=datetime_format, strict=False)
            for col in datetime_columns
        ]
    )


def drop_future_terminal_deals(df: pl.DataFrame) -> pl.DataFrame:
    now = datetime.now()
    filtered_df = df.filter(
        ~(
            (pl.col("DEAL_CREATEDATE") > now)
            | (
                pl.col("DEAL_CLOSED_WON_DATE").is_not_null()
                & (pl.col("DEAL_CLOSED_WON_DATE") > now)
            )
            | (
                pl.col("DEAL_DATETIME_ENTERED_CLOSEDLOST").is_not_null()
                & (pl.col("DEAL_DATETIME_ENTERED_CLOSEDLOST") > now)
            )
        )
    )

    return filtered_df


def drop_disqualified_deals(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(
        ~pl.col("DEAL_CLOSED_LOST_REASON").str.contains("Disqualified")
        | pl.col("DEAL_CLOSED_LOST_REASON").is_null()
    )
