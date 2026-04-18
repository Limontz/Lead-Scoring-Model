from enum import Enum
from typing import Any

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


class FunnelStage(str, Enum):
    CREATED = "DEAL_CREATEDATE"
    MQL = "DEAL_MQL_DATETIME"
    SQL = "DEAL_SQL_DATETIME"
    OPPORTUNITY = "DEAL_OPPORTUNITY_DATETIME"


class ModelName(str, Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    XGBOOST = "xgboost"


MODEL_REGISTRY: dict[ModelName, type[Any]] = {
    ModelName.LOGISTIC_REGRESSION: LogisticRegression,
    ModelName.XGBOOST: XGBClassifier,
}
