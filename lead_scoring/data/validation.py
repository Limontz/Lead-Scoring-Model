from dataclasses import dataclass
import polars as pl
import pandera.polars as pa
from datetime import datetime


@dataclass
class ValidationReport:
    row_count: int
    sql_without_mql: int
    opportunity_without_sql: int
    won_without_opportunity: int
    mql_before_created: int
    sql_before_mql: int
    opp_before_sql: int
    won_before_opp: int
    both_won_and_lost: int
    progressed_after_closed: int
    no_final_state: int
    inactive_deals: int
    future_closed_won_date: int
    future_closed_lost_date: int
    future_terminal_date: int
    future_created_date: int

    def hard_failures(self) -> dict[str, int]:
        return {
            "both_won_and_lost": self.both_won_and_lost,
            "mql_before_created": self.mql_before_created,
            "sql_before_mql": self.sql_before_mql,
            "opp_before_sql": self.opp_before_sql,
            "won_before_opp": self.won_before_opp,
            "sql_without_mql": self.sql_without_mql,
            "opportunity_without_sql": self.opportunity_without_sql,
            "won_without_opportunity": self.won_without_opportunity,
            "progressed_after_closed": self.progressed_after_closed,
        }

    def warnings(self) -> dict[str, int]:
        return {
            "no_final_state": self.no_final_state,
            "inactive_deals": self.inactive_deals,
            "future_closed_won_date": self.future_closed_won_date,
            "future_closed_lost_date": self.future_closed_lost_date,
            "future_terminal_date": self.future_terminal_date,
            "future_created_date": self.future_created_date,
        }

    def raise_if_invalid(self) -> None:
        failing = {k: v for k, v in self.hard_failures().items() if v > 0}
        if failing:
            raise ValueError(f"Typed validation failed: {failing}")


@pa.check_types
def build_validation_report(
    df: pl.DataFrame,
) -> ValidationReport:
    now = datetime.now()
    closed_at = pl.coalesce(
        [
            pl.col("DEAL_CLOSED_WON_DATE"),
            pl.col("DEAL_DATETIME_ENTERED_CLOSEDLOST"),
        ]
    )

    future_closed_won_date = df.filter(
        pl.col("DEAL_CLOSED_WON_DATE").is_not_null()
        & (pl.col("DEAL_CLOSED_WON_DATE") > now)
    ).height

    future_closed_lost_date = df.filter(
        pl.col("DEAL_DATETIME_ENTERED_CLOSEDLOST").is_not_null()
        & (pl.col("DEAL_DATETIME_ENTERED_CLOSEDLOST") > now)
    ).height

    future_terminal_date = df.filter(
        (
            pl.col("DEAL_CLOSED_WON_DATE").is_not_null()
            & (pl.col("DEAL_CLOSED_WON_DATE") > now)
        )
        | (
            pl.col("DEAL_DATETIME_ENTERED_CLOSEDLOST").is_not_null()
            & (pl.col("DEAL_DATETIME_ENTERED_CLOSEDLOST") > now)
        )
    ).height

    future_created = df.filter(pl.col("DEAL_CREATEDATE") > now).height

    return ValidationReport(
        row_count=df.height,
        sql_without_mql=df.filter(
            pl.col("DEAL_SQL_DATETIME").is_not_null()
            & pl.col("DEAL_MQL_DATETIME").is_null()
        ).height,
        opportunity_without_sql=df.filter(
            pl.col("DEAL_OPPORTUNITY_DATETIME").is_not_null()
            & pl.col("DEAL_SQL_DATETIME").is_null()
        ).height,
        won_without_opportunity=df.filter(
            pl.col("DEAL_CLOSED_WON_DATE").is_not_null()
            & pl.col("DEAL_OPPORTUNITY_DATETIME").is_null()
        ).height,
        mql_before_created=df.filter(
            pl.col("DEAL_MQL_DATETIME").is_not_null()
            & (pl.col("DEAL_MQL_DATETIME") < pl.col("DEAL_CREATEDATE"))
        ).height,
        sql_before_mql=df.filter(
            pl.col("DEAL_SQL_DATETIME").is_not_null()
            & pl.col("DEAL_MQL_DATETIME").is_not_null()
            & (pl.col("DEAL_SQL_DATETIME") < pl.col("DEAL_MQL_DATETIME"))
        ).height,
        opp_before_sql=df.filter(
            pl.col("DEAL_OPPORTUNITY_DATETIME").is_not_null()
            & pl.col("DEAL_SQL_DATETIME").is_not_null()
            & (pl.col("DEAL_OPPORTUNITY_DATETIME") < pl.col("DEAL_SQL_DATETIME"))
        ).height,
        won_before_opp=df.filter(
            pl.col("DEAL_CLOSED_WON_DATE").is_not_null()
            & pl.col("DEAL_OPPORTUNITY_DATETIME").is_not_null()
            & (pl.col("DEAL_CLOSED_WON_DATE") < pl.col("DEAL_OPPORTUNITY_DATETIME"))
        ).height,
        both_won_and_lost=df.filter(
            pl.col("DEAL_CLOSED_WON_DATE").is_not_null()
            & pl.col("DEAL_DATETIME_ENTERED_CLOSEDLOST").is_not_null()
        ).height,
        progressed_after_closed=df.filter(
            closed_at.is_not_null()
            & (
                (pl.col("DEAL_MQL_DATETIME") > closed_at)
                | (pl.col("DEAL_SQL_DATETIME") > closed_at)
                | (pl.col("DEAL_OPPORTUNITY_DATETIME") > closed_at)
            )
        ).height,
        no_final_state=df.filter(
            pl.col("DEAL_CLOSED_WON_DATE").is_null()
            & pl.col("DEAL_DATETIME_ENTERED_CLOSEDLOST").is_null()
        ).height,
        inactive_deals=df.filter(
            pl.col("DEAL_MQL_DATETIME").is_null()
            & pl.col("DEAL_SQL_DATETIME").is_null()
            & pl.col("DEAL_OPPORTUNITY_DATETIME").is_null()
            & pl.col("DEAL_CLOSED_WON_DATE").is_null()
            & pl.col("DEAL_DATETIME_ENTERED_CLOSEDLOST").is_null()
        ).height,
        future_closed_won_date=future_closed_won_date,
        future_closed_lost_date=future_closed_lost_date,
        future_terminal_date=future_terminal_date,
        future_created_date=future_created,
    )
