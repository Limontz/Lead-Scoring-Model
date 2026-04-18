import logging
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from lead_scoring.registry import MODEL_REGISTRY, ModelName
from lead_scoring.scoring_model.config import ScoringModelConfig

logger = logging.getLogger(__name__)


def build_preprocessing_pipeline(config: ScoringModelConfig) -> Any:
    if config.model_name == ModelName.XGBOOST:
        return "passthrough"

    transformers: list[tuple[str, Any, list[str]]] = []

    if config.categorical_features:
        transformers.append(
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop="if_binary",
                ),
                config.categorical_features,
            )
        )

    if config.numerical_features:
        transformers.append(("numeric", "passthrough", config.numerical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_model(config: ScoringModelConfig) -> Any:
    model_class = MODEL_REGISTRY[config.model_name]
    model_params = dict(config.model_params)
    return model_class(**model_params)


def build_model_pipeline(config: ScoringModelConfig) -> Pipeline:
    preprocessor = build_preprocessing_pipeline(config)
    model = build_model(config)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
