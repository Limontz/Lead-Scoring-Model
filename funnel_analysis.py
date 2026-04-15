from dataclasses import asdict
import json
import logging
import pandera as pa

from pandera.typing.polars import DataFrame

from lead_scoring.data.io import read_lead_data
from lead_scoring.data.manipulations import prepare_data_for_analysis
from lead_scoring.data.schema import RawDealsSchemaWithDatetime
from lead_scoring.data.validation import build_validation_report, cast_datetime_columns

@pa.check_types
def run_funnel_analysis_pipeline(
    path: str,
):
    df = read_lead_data(path)
    df = DataFrame[RawDealsSchemaWithDatetime](cast_datetime_columns(df))
    report = build_validation_report(df)
    report.raise_if_invalid()
    logging.info(
        "Data loaded and validated successfully.\nValidation report:\n%s",
        json.dumps(asdict(report), indent=2, sort_keys=False),
    )
    enriched_df = prepare_data_for_analysis(df)
    logging.info("Data enriched successfully.")
    return enriched_df, report

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    path = "./data/lead_data.csv"
    df, report = run_funnel_analysis_pipeline(path)
