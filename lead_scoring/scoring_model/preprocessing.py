import logging

import polars as pl
from lead_scoring.scoring_model.registry import FunnelStage
from pandera import polars as pa
from pandera.typing.polars import DataFrame

from lead_scoring.data.manipulations import prepare_data_for_analysis
from lead_scoring.data.schema import PreprocessedDealsSchema, RawDealsSchemaWithDatetime


logger = logging.getLogger(__name__)


def _filter_to_funnel_stage(df: pl.DataFrame, from_stage: FunnelStage) -> pl.DataFrame:
    stage_column = from_stage.value
    filtered_df = df.filter(pl.col(stage_column).is_not_null())
    return filtered_df


def _keep_resolved_deals(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("has_final_state"))


def _add_target_column(df: pl.DataFrame, target_column: str) -> pl.DataFrame:
    return df.with_columns(pl.col("is_closed_won").cast(pl.Int64).alias(target_column))


def _select_feature_and_target_columns(
    df: pl.DataFrame, feature: list[str], target: str
) -> pl.DataFrame:
    columns_to_select = feature + [target]
    return df.select(columns_to_select)


def _drop_null_values(df: pl.DataFrame, subset: list[str]) -> pl.DataFrame:
    null_count_exprs = [pl.col(column).null_count().alias(column) for column in subset]
    null_counts = df.select(null_count_exprs).to_dicts()[0]

    filtered_df = df.drop_nulls(subset=subset)
    dropped_rows_count = df.height - filtered_df.height

    logger.info(
        "Dropped %d rows with at least one null in selected columns; null counts by column: %s",
        dropped_rows_count,
        null_counts,
    )
    return filtered_df


@pa.check_types
def preprocess_data(
    df: DataFrame[RawDealsSchemaWithDatetime],
    features: list[str],
    target_column: str,
    from_stage: FunnelStage,
) -> DataFrame[PreprocessedDealsSchema]:

    enriched_df = prepare_data_for_analysis(df)
    enriched_df = _filter_to_funnel_stage(enriched_df, from_stage)
    enriched_df = _keep_resolved_deals(enriched_df)
    modeling_df = _add_target_column(enriched_df, target_column)
    modeling_df = _select_feature_and_target_columns(
        modeling_df, features, target_column
    )
    modeling_df = _drop_null_values(modeling_df, features)
    return DataFrame[PreprocessedDealsSchema](modeling_df)
