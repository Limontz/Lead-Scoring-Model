from datetime import datetime, timedelta

import polars as pl

from lead_scoring.registry import ModelName
from lead_scoring.scoring_model.training import stratified_train_test_split, train_model
from lead_scoring.tests.utils import build_test_scoring_config


def _build_training_df() -> pl.DataFrame:
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(12)]
    return pl.DataFrame(
        {
            "DEAL_CREATEDATE": dates,
            "DEAL_DEALSOURCE": ["Inbound", "Referral"] * 6,
            "DEAL_INDUSTRY": ["Tech", "Finance"] * 6,
            "CONTACT_ROLE": ["HR", "CEO"] * 6,
            "COMPANY_STATE": ["IT", "US"] * 6,
            "DEAL_SOURCE_DETAIL": ["Website", "Partner"] * 6,
            "UTM_SOURCE": ["google", "linkedin"] * 6,
            "LEAD_TYPE": ["demo", "trial"] * 6,
            "DEAL_HRIS_TECH_STACK": ["stack1", "stack2"] * 6,
            "DEAL_CCNL_MACRO": ["MacroA", "MacroB"] * 6,
            "target_closed_won": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )


def test_train_model_returns_pipeline() -> None:

    df = _build_training_df()
    config = build_test_scoring_config(model_name=ModelName.LOGISTIC_REGRESSION)

    X_train, X_test, y_train, y_test = stratified_train_test_split(
        df,
        target_col=config.target,
        train_portion=0.7,
        random_state=42,
    )

    pipeline = train_model(X_train, y_train, config)

    assert pipeline is not None
    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_train_model_with_custom_model_params() -> None:

    df = _build_training_df()
    config = build_test_scoring_config(
        model_name=ModelName.LOGISTIC_REGRESSION,
        model_params={"C": 0.1, "max_iter": 500},
    )

    X_train, X_test, y_train, y_test = stratified_train_test_split(
        df,
        target_col=config.target,
        train_portion=0.7,
        random_state=42,
    )

    pipeline = train_model(X_train, y_train, config)
    model = pipeline.named_steps["model"]

    assert model.C == 0.1
    assert model.max_iter == 500
