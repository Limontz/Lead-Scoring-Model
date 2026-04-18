import logging
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from lead_scoring.registry import MODEL_REGISTRY
from lead_scoring.scoring_model.config import ScoringModelConfig

logger = logging.getLogger(__name__)


def build_preprocessing_pipeline(
    categorical_features: list[str],
) -> ColumnTransformer:

    return ColumnTransformer(
        transformers=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop="if_binary",
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def build_model(config: ScoringModelConfig) -> Any:
    model_class = MODEL_REGISTRY[config.model_name]
    model_params = dict(config.model_params)
    return model_class(**model_params)


def build_model_pipeline(config: ScoringModelConfig) -> Pipeline:

    preprocessor = build_preprocessing_pipeline(config.categorical_features)
    model = build_model(config)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline
