from datetime import datetime, timedelta

import polars as pl
from sklearn.linear_model import LogisticRegression

from lead_scoring.scoring_model.config import ScoringModelConfig
from lead_scoring.scoring_model.training import (
    build_logistic_regression_model,
    train_logistic_regression,
)


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


def test_build_logistic_regression_model() -> None:
    config = ScoringModelConfig(random_state=7)

    model = build_logistic_regression_model(config)

    assert isinstance(model, LogisticRegression)
    assert model.random_state == 7
    assert model.max_iter == 1000


def test_train_logistic_regression() -> None:
    df = _build_training_df()
    config = ScoringModelConfig()

    pipeline, metrics = train_logistic_regression(df=df, config=config)

    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps
    assert 0.0 <= metrics.roc_auc <= 1.0
    assert 0.0 <= metrics.average_precision <= 1.0
    assert 0.0 <= metrics.accuracy <= 1.0
