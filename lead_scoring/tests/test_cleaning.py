from datetime import datetime, timedelta

import polars as pl

from lead_scoring.data.cleaning import (
    cast_datetime_columns,
    drop_disqualified_deals,
    drop_future_terminal_deals,
)


def test_cast_datetime_columns() -> None:
    df = pl.DataFrame(
        {
            "DEAL_CREATEDATE": ["2024-01-10 10:30:00"],
            "DEAL_MQL_DATETIME": ["2024-01-11 10:30:00"],
            "DEAL_SQL_DATETIME": ["2024-01-12 10:30:00"],
            "DEAL_OPPORTUNITY_DATETIME": ["2024-01-13 10:30:00"],
            "DEAL_CLOSED_WON_DATE": ["2024-01-14 10:30:00"],
            "DEAL_DATETIME_ENTERED_CLOSEDLOST": ["2024-01-15 10:30:00"],
        }
    )

    cleaned = cast_datetime_columns(df)

    assert cleaned.schema["DEAL_CREATEDATE"] == pl.Datetime
    assert cleaned["DEAL_CREATEDATE"].to_list()[0] == datetime(2024, 1, 10, 10, 30, 0)


def test_drop_future_terminal_deals() -> None:
    now = datetime.now()
    past = now - timedelta(days=10)
    future = now + timedelta(days=10)

    df = pl.DataFrame(
        {
            "DEAL_ID": [1, 2, 3, 4],
            "DEAL_CREATEDATE": [past, future, past, past],
            "DEAL_CLOSED_WON_DATE": [None, None, future, None],
            "DEAL_DATETIME_ENTERED_CLOSEDLOST": [None, None, None, past],
        }
    )

    cleaned = drop_future_terminal_deals(df)

    assert cleaned["DEAL_ID"].to_list() == [1, 4]


def test_drop_disqualified_deals_filters_disqualified_reason() -> None:
    df = pl.DataFrame(
        {
            "DEAL_ID": [1, 2, 3, 4],
            "DEAL_CLOSED_LOST_REASON": [
                "Disqualified - not a fit",
                "Disqualified - budget constraints",
                None,
                "Budget constraints",
            ],
        }
    )

    cleaned = drop_disqualified_deals(df)

    assert cleaned["DEAL_ID"].to_list() == [3, 4]
