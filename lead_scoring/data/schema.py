from datetime import datetime

import pandera.polars as pa
from pandera.typing.polars import Series


class PanderaBaseModel(pa.DataFrameModel):
    class Config:
        strict = True


class RawDealsSchema(PanderaBaseModel):
    DEAL_ID: Series[str] = pa.Field(nullable=False, unique=True)

    DEAL_CREATEDATE: Series[str] = pa.Field(nullable=False)
    DEAL_MQL_DATETIME: Series[str] = pa.Field(nullable=True)
    DEAL_SQL_DATETIME: Series[str] = pa.Field(nullable=True)
    DEAL_OPPORTUNITY_DATETIME: Series[str] = pa.Field(nullable=True)
    DEAL_CLOSED_WON_DATE: Series[str] = pa.Field(nullable=True)
    DEAL_DATETIME_ENTERED_CLOSEDLOST: Series[str] = pa.Field(nullable=True)

    DEAL_STAGE_ID: Series[str] = pa.Field(nullable=False)
    DEAL_PIPELINE_ID: Series[str] = pa.Field(nullable=False)

    DEAL_AMOUNT: Series[float] = pa.Field(ge=0, nullable=False)
    DEAL_NUMBER_OF_EMPLOYEES: Series[int] = pa.Field(ge=0, nullable=False)
    DEAL_NUMERO_CEDOLINI: Series[int] = pa.Field(ge=0, nullable=False)
    DEAL_NUMBER_TIMES_CONTACTED: Series[int] = pa.Field(ge=0, nullable=True)

    DEAL_OWNER_ID: Series[str] = pa.Field(nullable=True)
    DEAL_BDR_OWNER_ID: Series[str] = pa.Field(nullable=True)

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
    COMPANY_REVENUE: Series[float] = pa.Field(ge=0, nullable=True)


class TypedDealsSchema(PanderaBaseModel):
    DEAL_ID: Series[str] = pa.Field(nullable=False, unique=True)

    DEAL_CREATEDATE: Series[datetime] = pa.Field(nullable=False)
    DEAL_MQL_DATETIME: Series[datetime] = pa.Field(nullable=True)
    DEAL_SQL_DATETIME: Series[datetime] = pa.Field(nullable=True)
    DEAL_OPPORTUNITY_DATETIME: Series[datetime] = pa.Field(nullable=True)
    DEAL_CLOSED_WON_DATE: Series[datetime] = pa.Field(nullable=True)
    DEAL_DATETIME_ENTERED_CLOSEDLOST: Series[datetime] = pa.Field(nullable=True)

    DEAL_STAGE_ID: Series[str] = pa.Field(nullable=True)
    DEAL_PIPELINE_ID: Series[str] = pa.Field(nullable=True)

    DEAL_AMOUNT: Series[float] = pa.Field(ge=0, nullable=True)
    DEAL_NUMBER_OF_EMPLOYEES: Series[int] = pa.Field(ge=0, nullable=True)
    DEAL_NUMERO_CEDOLINI: Series[int] = pa.Field(ge=0, nullable=True)
    DEAL_NUMBER_TIMES_CONTACTED: Series[int] = pa.Field(ge=0, nullable=True)

    DEAL_OWNER_ID: Series[str] = pa.Field(nullable=True)
    DEAL_BDR_OWNER_ID: Series[str] = pa.Field(nullable=True)

    DEAL_INDUSTRY: Series[str] = pa.Field(nullable=True)
    DEAL_DEALSOURCE: Series[str] = pa.Field(nullable=True)
    DEAL_SOURCE_DETAIL: Series[str] = pa.Field(nullable=True)
    DEAL_CCNL_MACRO: Series[str] = pa.Field(nullable=True)
    DEAL_CCNL_DETAIL: Series[str] = pa.Field(nullable=True)
    DEAL_MODULI_AGGIUNTIVI_ACQUISTATI: Series[str] = pa.Field(nullable=True)
    UTM_SOURCE: Series[str] = pa.Field(nullable=True)
    LEAD_TYPE: Series[str] = pa.Field(nullable=True)
    DEAL_HRIS_TECH_STACK: Series[str] = pa.Field(nullable=True)
    CONTACT_ROLE: Series[str] = pa.Field(nullable=True)
    COMPANY_STATE: Series[str] = pa.Field(nullable=True)

    COMPANY_FUNDING_YEAR: Series[int] = pa.Field(nullable=True)
    COMPANY_REVENUE: Series[float] = pa.Field(ge=0, nullable=True)