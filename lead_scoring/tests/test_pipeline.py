from datetime import datetime, timedelta

import polars as pl
from sklearn.linear_model import LogisticRegression

from lead_scoring.scoring_model.config import ScoringModelConfig
from lead_scoring.scoring_model.pipeline import (
    build_model_pipeline,
    stratified_train_test_split,
)


def _build_test_df() -> pl.DataFrame:
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)]
    return pl.DataFrame(
        {
            "DEAL_CREATEDATE": dates,
            "DEAL_DEALSOURCE": ["Inbound", "Referral"] * 5,
            "DEAL_INDUSTRY": ["Tech", "Finance"] * 5,
            "CONTACT_ROLE": ["HR", "CEO"] * 5,
            "COMPANY_STATE": ["IT", "US"] * 5,
            "DEAL_SOURCE_DETAIL": ["Website", "Partner"] * 5,
            "UTM_SOURCE": ["google", "linkedin"] * 5,
            "LEAD_TYPE": ["demo", "trial"] * 5,
            "DEAL_HRIS_TECH_STACK": ["stack1", "stack2"] * 5,
            "DEAL_CCNL_MACRO": ["MacroA", "MacroB"] * 5,
            "target_closed_won": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )


def test_stratified_train_test_split() -> None:
    df = _build_test_df()
    config = ScoringModelConfig()

    X_train, X_test, y_train, y_test = stratified_train_test_split(
        df,
        target_col=config.target,
        train_portion=0.7,
        random_state=42,
    )

    assert len(X_train) == 7
    assert len(X_test) == 3
    assert 0 in y_train.to_list() and 1 in y_train.to_list()
    assert 0 in y_test.to_list() and 1 in y_test.to_list()


def test_build_model_pipeline_with_logistic_regression() -> None:
    config = ScoringModelConfig()
    model = LogisticRegression(random_state=42, max_iter=200)

    pipeline = build_model_pipeline(model, config)

    assert pipeline is not None
    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps


def test_pipeline_fit_predict_logistic_regression() -> None:
    df = _build_test_df()
    config = ScoringModelConfig()

    X_train, X_test, y_train, y_test = stratified_train_test_split(
        df,
        target_col=config.target,
        train_portion=0.7,
        random_state=42,
    )

    model = LogisticRegression(random_state=42, max_iter=200)
    pipeline = build_model_pipeline(model, config)

    X_train_pd = X_train.to_pandas()
    X_test_pd = X_test.to_pandas()
    y_train_np = y_train.to_numpy()

    pipeline.fit(X_train_pd, y_train_np)
    preds = pipeline.predict(X_test_pd)

    assert len(preds) == len(X_test_pd)
    assert all(p in [0, 1] for p in preds)
