from datetime import datetime

import polars as pl

from lead_scoring.data.manipulations import prepare_data_for_analysis
from lead_scoring.registry import FunnelStage
from lead_scoring.scoring_model.preprocessing import (
    _add_target_column,
    _drop_null_values,
    _filter_to_funnel_stage,
    _keep_resolved_deals,
    _select_feature_and_target_columns,
    preprocess_data,
)


def _build_raw_deals_df() -> pl.DataFrame:
    created = datetime(2024, 1, 1, 9, 0, 0)
    mql = datetime(2024, 1, 2, 9, 0, 0)
    sql = datetime(2024, 1, 3, 9, 0, 0)
    opp = datetime(2024, 1, 4, 9, 0, 0)
    won = datetime(2024, 1, 5, 9, 0, 0)
    lost = datetime(2024, 1, 6, 9, 0, 0)

    return pl.DataFrame(
        {
            "DEAL_ID": [1, 2, 3, 4],
            "DEAL_CREATEDATE": [created, created, created, created],
            "DEAL_MQL_DATETIME": [mql, mql, mql, None],
            "DEAL_SQL_DATETIME": [sql, sql, sql, None],
            "DEAL_OPPORTUNITY_DATETIME": [opp, opp, opp, None],
            "DEAL_CLOSED_WON_DATE": [won, None, None, won],
            "DEAL_DATETIME_ENTERED_CLOSEDLOST": [None, lost, None, None],
            "DEAL_CLOSED_LOST_REASON": [None, "pricing", None, None],
            "DEAL_CASO_USO_PAYROLL": ["yes", "yes", "yes", "yes"],
            "DEAL_PIPELINE_ID": ["p1", "p1", "p1", "p1"],
            "DEAL_AMOUNT": [1000.0, 2000.0, 1500.0, 800.0],
            "DEAL_NUMBER_OF_EMPLOYEES": [10, 20, 30, 40],
            "DEAL_NUMERO_CEDOLINI": [2, 3, 4, 1],
            "DEAL_NUMBER_TIMES_CONTACTED": [1, 2, 3, 1],
            "DEAL_OWNER_ID": [101, 102, 103, 104],
            "DEAL_BDR_OWNER_ID": [201, 202, 203, 204],
            "DEAL_INDUSTRY": ["Tech", "Finance", "Retail", "Tech"],
            "DEAL_DEALSOURCE": ["Inbound", "Referral", "Inbound", "Outbound"],
            "DEAL_SOURCE_DETAIL": ["Website", "Partner", "Website", "Call"],
            "DEAL_CCNL_MACRO": ["MacroA", "MacroB", "MacroA", "MacroA"],
            "DEAL_CCNL_DETAIL": ["Detail1", "Detail2", "Detail3", "Detail4"],
            "DEAL_MODULI_AGGIUNTIVI_ACQUISTATI": ["module_a", None, None, None],
            "UTM_SOURCE": ["google", "linkedin", "google", "email"],
            "LEAD_TYPE": ["demo", "demo", "trial", "demo"],
            "DEAL_HRIS_TECH_STACK": ["stack1", "stack2", "stack3", "stack1"],
            "CONTACT_ROLE": ["HR", "CEO", "HR", "COO"],
            "COMPANY_STATE": ["IT", "IT", "IT", "IT"],
            "DEAL_DEMO_SCORE": [70, 80, 90, 60],
            "COMPANY_FUNDING_YEAR": [2020, 2019, 2021, 2018],
            "COMPANY_REVENUE": [1000000, 2000000, 1500000, 800000],
        }
    )


def test_filter_to_funnel_stage() -> None:
    df = pl.DataFrame(
        {
            "DEAL_ID": [1, 2],
            "DEAL_MQL_DATETIME": [datetime(2024, 1, 1), None],
        }
    )

    filtered = _filter_to_funnel_stage(df, FunnelStage.MQL)

    assert filtered["DEAL_ID"].to_list() == [1]


def test_filter_to_funnel_stage_sql() -> None:
    df = pl.DataFrame(
        {
            "DEAL_ID": [1, 2, 3],
            "DEAL_SQL_DATETIME": [datetime(2024, 1, 1), None, datetime(2024, 1, 3)],
        }
    )

    filtered = _filter_to_funnel_stage(df, FunnelStage.SQL)

    assert filtered["DEAL_ID"].to_list() == [1, 3]


def test_keep_resolved_deals() -> None:
    df = pl.DataFrame({"DEAL_ID": [1, 2, 3], "has_final_state": [True, False, True]})

    resolved = _keep_resolved_deals(df)

    assert resolved["DEAL_ID"].to_list() == [1, 3]


def test_add_target_column() -> None:
    df = pl.DataFrame({"is_closed_won": [True, False]})

    transformed = _add_target_column(df, "target_closed_won")

    assert transformed["target_closed_won"].to_list() == [1, 0]


def test_select_feature_and_target_columns() -> None:
    df = pl.DataFrame(
        {
            "feature_a": [1, 2],
            "feature_b": [3, 4],
            "target": [0, 1],
            "unused": [9, 9],
        }
    )

    selected = _select_feature_and_target_columns(
        df, ["feature_a", "feature_b"], "target"
    )

    assert selected.columns == ["feature_a", "feature_b", "target"]


def test_drop_null_values() -> None:
    df = pl.DataFrame(
        {
            "feature_a": [1, None, 3],
            "feature_b": [10, 20, None],
            "target": [1, 0, 1],
        }
    )

    cleaned = _drop_null_values(df, subset=["feature_a", "feature_b"])

    assert cleaned.to_dicts() == [{"feature_a": 1, "feature_b": 10, "target": 1}]


def test_preprocess_data_integration() -> None:
    df = _build_raw_deals_df()
    features = [
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

    result = preprocess_data(
        df=df,
        total_features=features,
        target_column="target_closed_won",
        from_stage=FunnelStage.MQL,
    )

    assert result.columns == [*features, "target_closed_won"]
    assert result.height == 2
    assert result["target_closed_won"].to_list() == [1, 0]


def test_prepare_data_for_analysis() -> None:
    df = _build_raw_deals_df()

    enriched = prepare_data_for_analysis(df)

    assert "has_final_state" in enriched.columns
    assert "final_status" in enriched.columns
    assert "creation_to_mql_days" in enriched.columns
    assert enriched["has_final_state"].to_list() == [True, True, False, True]
    assert enriched["final_status"].to_list() == ["won", "lost", "open", "won"]
