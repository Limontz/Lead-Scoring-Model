from lead_scoring.registry import ModelName
from lead_scoring.scoring_model.config import ScoringModelConfig

TEST_TARGET = "target_closed_won"
TEST_CATEGORICAL_FEATURES = [
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


def build_test_scoring_config(
    *,
    model_name: ModelName = ModelName.LOGISTIC_REGRESSION,
    model_params: dict | None = None,
    categorical_features: list[str] | None = None,
    numerical_features: list[str] | None = None,
) -> ScoringModelConfig:
    resolved_categorical_features = (
        list(TEST_CATEGORICAL_FEATURES)
        if categorical_features is None
        else list(categorical_features)
    )
    resolved_numerical_features = (
        [] if numerical_features is None else list(numerical_features)
    )
    resolved_model_params = {} if model_params is None else dict(model_params)

    return ScoringModelConfig(
        target=TEST_TARGET,
        categorical_features=resolved_categorical_features,
        numerical_features=resolved_numerical_features,
        model_name=model_name,
        model_params=resolved_model_params,
        training_set_portion=0.7,
        random_state=42,
    )
