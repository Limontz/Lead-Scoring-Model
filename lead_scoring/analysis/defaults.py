import polars as pl


def get_step_config() -> dict[str, dict[str, str | pl.Expr | None]]:
    return {
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


def get_numeric_features_default() -> list[str]:
    return [
        "DEAL_DEMO_SCORE",
        "DEAL_AMOUNT",
        "DEAL_NUMBER_OF_EMPLOYEES",
        "DEAL_NUMERO_CEDOLINI",
        "DEAL_NUMBER_TIMES_CONTACTED",
        "COMPANY_REVENUE",
        "COMPANY_FUNDING_YEAR",
    ]


def get_categorical_features_default() -> list[str]:
    return [
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


def get_duration_features_default() -> list[str]:
    return [
        "creation_to_mql_days",
        "mql_to_sql_days",
        "sql_to_opp_days",
        "opp_to_won_days",
        "opp_to_lost_days",
        "creation_to_won_days",
        "creation_to_lost_days",
        "creation_to_closed_days",
    ]


def get_stage_month_cols() -> dict[str, str]:
    return {
        "created": "created_month",
        "mql": "mql_month",
        "sql": "sql_month",
        "opportunity": "opportunity_month",
        "closed_won": "closed_won_month",
        "closed_lost": "closed_lost_month",
    }


def get_temporal_conversion_cols_default() -> list[str]:
    return [
        "cr_created_to_mql",
        "cr_mql_to_sql",
        "cr_sql_to_opp",
        "cr_opp_to_won",
        "cr_created_to_won",
    ]


def get_temporal_duration_cols_default() -> list[str]:
    return [
        "creation_to_mql_days",
        "mql_to_sql_days",
        "sql_to_opp_days",
        "opp_to_won_days",
        "opp_to_lost_days",
        "creation_to_won_days",
        "creation_to_lost_days",
        "creation_to_closed_days",
    ]


def get_stage_order() -> list[str]:
    return ["created", "mql", "sql", "opportunity", "closed_won", "closed_lost"]


def get_funnel_stage_date_cols() -> dict[str, str]:
    return {
        "created": "DEAL_CREATEDATE",
        "mql": "DEAL_MQL_DATETIME",
        "sql": "DEAL_SQL_DATETIME",
        "opportunity": "DEAL_OPPORTUNITY_DATETIME",
        "closed_won": "DEAL_CLOSED_WON_DATE",
        "closed_lost": "DEAL_DATETIME_ENTERED_CLOSEDLOST",
    }
