from enum import Enum


class FunnelStage(str, Enum):
    CREATED = "DEAL_CREATEDATE"
    MQL = "DEAL_MQL_DATETIME"
    SQL = "DEAL_SQL_DATETIME"
    OPPORTUNITY = "DEAL_OPPORTUNITY_DATETIME"
