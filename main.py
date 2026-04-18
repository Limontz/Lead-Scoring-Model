import logging

from lead_scoring.data.cleaning import clean_data
from lead_scoring.data.io import read_lead_data
from lead_scoring.scoring_model.config import get_scoring_model_config
from lead_scoring.scoring_model.preprocessing import preprocess_data
from lead_scoring.scoring_model.training import (
    stratified_train_test_split,
    train_model,
)
from lead_scoring.scoring_model.evaluation import (
    evaluate_binary_classifier,
    get_linear_model_weights,
)
from lead_scoring.registry import FunnelStage


def run_full_workflow(
    data_path: str,
    from_stage: FunnelStage,
):
    model_config = get_scoring_model_config()

    lead_score_df = read_lead_data(data_path)
    clean_lead_score_df = clean_data(lead_score_df)
    preprocessed_lead_score_df = preprocess_data(
        clean_lead_score_df,
        model_config.total_features,
        model_config.target,
        from_stage=from_stage,
    )

    X_train, X_test, y_train, y_test = stratified_train_test_split(
        df=preprocessed_lead_score_df,
        target_col=model_config.target,
        train_portion=model_config.training_set_portion,
        random_state=model_config.random_state,
    )

    pipeline = train_model(X_train, y_train, model_config)

    metrics = evaluate_binary_classifier(pipeline, X_test, y_test, model_config)

    return pipeline, metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    pipeline, metrics = run_full_workflow(
        "data/lead_data.csv",
        from_stage=FunnelStage.MQL,
    )
    print("Model performance:")
    print(f"  ROC-AUC: {metrics.roc_auc:.8f}")
    print(f"  Average Precision: {metrics.average_precision:.8f}")
    print(f"  Accuracy: {metrics.accuracy:.8f}")

    try:
        weights_df = get_linear_model_weights(pipeline)
        print("\nTop linear model weights (by absolute value):")
        print(weights_df.select(["feature", "weight"]).head(20))
    except ValueError:
        print("\nModel does not expose linear coefficients (e.g., tree-based model).")
