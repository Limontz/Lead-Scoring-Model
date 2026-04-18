import logging

from lead_scoring.data.cleaning import clean_data
from lead_scoring.data.io import read_lead_data
from lead_scoring.scoring_model.config import get_scoring_model_config
from lead_scoring.scoring_model.preprocessing import preprocess_data
from lead_scoring.registry import FunnelStage


def score_lead(path: str, from_stage: FunnelStage):
    model_config = get_scoring_model_config()
    lead_score_df = read_lead_data(path)
    clean_lead_score_df = clean_data(lead_score_df)
    preprocessed_lead_score_df = preprocess_data(
        clean_lead_score_df,
        model_config.features,
        model_config.target,
        from_stage=from_stage,
    )
    return preprocessed_lead_score_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    lead_score_df = score_lead("data/lead_data.csv", from_stage=FunnelStage.SQL)
