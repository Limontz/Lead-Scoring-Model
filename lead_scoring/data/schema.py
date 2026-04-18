from datetime import datetime
import polars as pl

import pandera.polars as pa
from pandera.typing.polars import Series


class PanderaBaseModel(pa.DataFrameModel):
    class Config:  # pyright: ignore[reportIncompatibleVariableOverride]
        strict = True


class RawDealsSchema(PanderaBaseModel):
    DEAL_ID: Series[pl.Int64] = pa.Field(nullable=False, unique=True)

    DEAL_CREATEDATE: Series[str] = pa.Field(nullable=False)
    DEAL_MQL_DATETIME: Series[str] = pa.Field(nullable=True)
    DEAL_SQL_DATETIME: Series[str] = pa.Field(nullable=True)
    DEAL_OPPORTUNITY_DATETIME: Series[str] = pa.Field(nullable=True)
    DEAL_CLOSED_WON_DATE: Series[str] = pa.Field(nullable=True)
    DEAL_DATETIME_ENTERED_CLOSEDLOST: Series[str] = pa.Field(nullable=True)

    DEAL_CLOSED_LOST_REASON: Series[str] = pa.Field(nullable=True)
    DEAL_CASO_USO_PAYROLL: Series[str] = pa.Field(nullable=False)
    DEAL_PIPELINE_ID: Series[str] = pa.Field(nullable=False)

    DEAL_AMOUNT: Series[float] = pa.Field(ge=0, nullable=False)
    DEAL_NUMBER_OF_EMPLOYEES: Series[int] = pa.Field(ge=0, nullable=False)
    DEAL_NUMERO_CEDOLINI: Series[int] = pa.Field(ge=0, nullable=False)
    DEAL_NUMBER_TIMES_CONTACTED: Series[int] = pa.Field(ge=0, nullable=True)

    DEAL_OWNER_ID: Series[pl.Int64] = pa.Field(nullable=True)
    DEAL_BDR_OWNER_ID: Series[pl.Int64] = pa.Field(nullable=True)

    DEAL_INDUSTRY: Series[str] = pa.Field(nullable=False)
    DEAL_DEALSOURCE: Series[str] = pa.Field(nullable=False)
    DEAL_SOURCE_DETAIL: Series[str] = pa.Field(nullable=True)
    DEAL_CCNL_MACRO: Series[str] = pa.Field(nullable=True)
    DEAL_CCNL_DETAIL: Series[str] = pa.Field(nullable=False)
    DEAL_MODULI_AGGIUNTIVI_ACQUISTATI: Series[str] = pa.Field(nullable=True)
    UTM_SOURCE: Series[str] = pa.Field(nullable=True)
    LEAD_TYPE: Series[str] = pa.Field(nullable=True)
    DEAL_HRIS_TECH_STACK: Series[str] = pa.Field(nullable=False)
    CONTACT_ROLE: Series[str] = pa.Field(nullable=False)
    COMPANY_STATE: Series[str] = pa.Field(nullable=False)
    DEAL_DEMO_SCORE: Series[int] = pa.Field(nullable=True)

    COMPANY_FUNDING_YEAR: Series[int] = pa.Field(nullable=True)
    COMPANY_REVENUE: Series[pl.Int64] = pa.Field(ge=0, nullable=True)


class RawDealsSchemaWithDatetime(PanderaBaseModel):
    DEAL_ID: Series[pl.Int64] = pa.Field(nullable=False, unique=True)

    DEAL_CREATEDATE: Series[datetime] = pa.Field(nullable=False)
    DEAL_MQL_DATETIME: Series[datetime] = pa.Field(nullable=True)
    DEAL_SQL_DATETIME: Series[datetime] = pa.Field(nullable=True)
    DEAL_OPPORTUNITY_DATETIME: Series[datetime] = pa.Field(nullable=True)
    DEAL_CLOSED_WON_DATE: Series[datetime] = pa.Field(nullable=True)
    DEAL_DATETIME_ENTERED_CLOSEDLOST: Series[datetime] = pa.Field(nullable=True)

    DEAL_CLOSED_LOST_REASON: Series[str] = pa.Field(nullable=True)
    DEAL_CASO_USO_PAYROLL: Series[str] = pa.Field(nullable=False)
    DEAL_PIPELINE_ID: Series[str] = pa.Field(nullable=False)

    DEAL_AMOUNT: Series[float] = pa.Field(ge=0, nullable=False)
    DEAL_NUMBER_OF_EMPLOYEES: Series[int] = pa.Field(ge=0, nullable=False)
    DEAL_NUMERO_CEDOLINI: Series[int] = pa.Field(ge=0, nullable=False)
    DEAL_NUMBER_TIMES_CONTACTED: Series[int] = pa.Field(ge=0, nullable=True)

    DEAL_OWNER_ID: Series[pl.Int64] = pa.Field(nullable=True)
    DEAL_BDR_OWNER_ID: Series[pl.Int64] = pa.Field(nullable=True)

    DEAL_INDUSTRY: Series[str] = pa.Field(nullable=False)
    DEAL_DEALSOURCE: Series[str] = pa.Field(nullable=False)
    DEAL_SOURCE_DETAIL: Series[str] = pa.Field(nullable=True)
    DEAL_CCNL_MACRO: Series[str] = pa.Field(nullable=True)
    DEAL_CCNL_DETAIL: Series[str] = pa.Field(nullable=False)
    DEAL_MODULI_AGGIUNTIVI_ACQUISTATI: Series[str] = pa.Field(nullable=True)
    UTM_SOURCE: Series[str] = pa.Field(nullable=True)
    LEAD_TYPE: Series[str] = pa.Field(nullable=True)
    DEAL_HRIS_TECH_STACK: Series[str] = pa.Field(nullable=False)
    CONTACT_ROLE: Series[str] = pa.Field(nullable=False)
    COMPANY_STATE: Series[str] = pa.Field(nullable=False)
    DEAL_DEMO_SCORE: Series[int] = pa.Field(nullable=True)

    COMPANY_FUNDING_YEAR: Series[int] = pa.Field(nullable=True)
    COMPANY_REVENUE: Series[pl.Int64] = pa.Field(ge=0, nullable=True)


class EnrichedDealsSchema(RawDealsSchemaWithDatetime):
    is_mql: Series[bool] = pa.Field(nullable=False)
    is_sql: Series[bool] = pa.Field(nullable=False)
    is_opportunity: Series[bool] = pa.Field(nullable=False)
    is_closed_won: Series[bool] = pa.Field(nullable=False)
    is_closed_lost: Series[bool] = pa.Field(nullable=False)

    has_final_state: Series[bool] = pa.Field(nullable=False)
    is_open: Series[bool] = pa.Field(nullable=False)
    is_inactive: Series[bool] = pa.Field(nullable=False)
    final_status: Series[str] = pa.Field(nullable=False)
    closed_at: Series[datetime] = pa.Field(nullable=True)

    created_month: Series[datetime] = pa.Field(nullable=True)
    mql_month: Series[datetime] = pa.Field(nullable=True)
    sql_month: Series[datetime] = pa.Field(nullable=True)
    opportunity_month: Series[datetime] = pa.Field(nullable=True)
    closed_won_month: Series[datetime] = pa.Field(nullable=True)
    closed_lost_month: Series[datetime] = pa.Field(nullable=True)

    creation_to_mql_days: Series[int] = pa.Field(nullable=True, ge=0)
    mql_to_sql_days: Series[int] = pa.Field(nullable=True, ge=0)
    sql_to_opp_days: Series[int] = pa.Field(nullable=True, ge=0)
    opp_to_won_days: Series[int] = pa.Field(nullable=True, ge=0)
    opp_to_lost_days: Series[int] = pa.Field(nullable=True, ge=0)

    creation_to_sql_days: Series[int] = pa.Field(nullable=True, ge=0)
    creation_to_opp_days: Series[int] = pa.Field(nullable=True, ge=0)
    creation_to_won_days: Series[int] = pa.Field(nullable=True, ge=0)
    creation_to_lost_days: Series[int] = pa.Field(nullable=True, ge=0)
    creation_to_closed_days: Series[int] = pa.Field(nullable=True, ge=0)


class PreprocessedDealsSchema(PanderaBaseModel):
    DEAL_DEALSOURCE: Series[str] = pa.Field(nullable=False)
    DEAL_SOURCE_DETAIL: Series[str] = pa.Field(nullable=True)
    UTM_SOURCE: Series[str] = pa.Field(nullable=True)
    LEAD_TYPE: Series[str] = pa.Field(nullable=True)
    DEAL_INDUSTRY: Series[str] = pa.Field(nullable=False)
    CONTACT_ROLE: Series[str] = pa.Field(nullable=False)
    COMPANY_STATE: Series[str] = pa.Field(nullable=False)
    DEAL_HRIS_TECH_STACK: Series[str] = pa.Field(nullable=False)
    DEAL_CCNL_MACRO: Series[str] = pa.Field(nullable=True)

    target_closed_won: Series[pl.Int64] = pa.Field(nullable=False, isin=[0, 1])
