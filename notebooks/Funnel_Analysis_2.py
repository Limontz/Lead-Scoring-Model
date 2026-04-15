import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from datetime import datetime
    import polars as pl
    from dataclasses import asdict
    import json
    import logging
    import pandera as pa

    from pandera.typing.polars import DataFrame

    from lead_scoring.data.io import read_lead_data
    from lead_scoring.data.manipulations import (
        cast_datetime_columns,
        prepare_data_for_analysis,
        drop_future_terminal_deals,
    )
    from lead_scoring.data.schema import RawDealsSchemaWithDatetime
    from lead_scoring.data.validation import build_validation_report

    from lead_scoring.analysis.funnel_analysis import analyze_funnel

    return (
        analyze_funnel,
        asdict,
        build_validation_report,
        cast_datetime_columns,
        drop_future_terminal_deals,
        json,
        logging,
        mo,
        prepare_data_for_analysis,
        read_lead_data,
    )


@app.cell
def _(
    analyze_funnel,
    asdict,
    build_validation_report,
    cast_datetime_columns,
    drop_future_terminal_deals,
    json,
    logging,
    prepare_data_for_analysis,
    read_lead_data,
):
    path = "./data/lead_data.csv"

    df = read_lead_data(path)
    df = cast_datetime_columns(df)

    report = build_validation_report(df)
    report.raise_if_invalid()
    print(
        "Data loaded and validated successfully.\nValidation report:\n%s",
        json.dumps(asdict(report), indent=2, sort_keys=False),
    )
    df = drop_future_terminal_deals(df)
    enriched_df = prepare_data_for_analysis(df)
    logging.info("Data enriched successfully.")

    result = analyze_funnel(enriched_df)
    return enriched_df, result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Static Analysis
    """)
    return


@app.cell
def _(result):
    result.plot_stage_volumes().show()
    result.plot_step_conversion_rates().show()
    result.plot_vs_created_conversion_rates().show()

    # print("sql_to_demo_rate:", result.process_metrics.sql_to_demo_score_rate)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Step-Focused Analysis
    """)
    return


@app.cell
def _(analyze_funnel, enriched_df):
    opp_result = analyze_funnel(
        enriched_df,
        step="sql_to_opp",
    )

    numeric_columns_to_analyse = [
        "DEAL_DEMO_SCORE",
        "DEAL_AMOUNT",
        "DEAL_NUMBER_TIMES_CONTACTED",
        "COMPANY_REVENUE",
    ]

    categorical_columns_to_analyse = [
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

    for col in numeric_columns_to_analyse:
        opp_result.plot_numeric_boxplot(col).show()

    for col in categorical_columns_to_analyse:
        opp_result.plot_categorical_conversion(col).show()

    opp_result.plot_closed_lost_reasons().show()
    return categorical_columns_to_analyse, col, numeric_columns_to_analyse


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Segment Analysis
    """)
    return


@app.cell
def _(analyze_funnel, enriched_df):
    segment_result = analyze_funnel(
        enriched_df,
        segment_col="LEAD_TYPE",
        step="opp_to_won",
    )

    segment_result.plot_stage_volumes().show()
    segment_result.plot_step_conversion_rates().show()
    segment_result.plot_vs_created_conversion_rates().show()
    return (segment_result,)


@app.cell
def _(
    categorical_columns_to_analyse,
    col,
    numeric_columns_to_analyse,
    segment_result,
):
    for segment_col in numeric_columns_to_analyse:
        segment_result.plot_numeric_boxplot(col).show()

    for segment_col in categorical_columns_to_analyse:
        segment_result.plot_categorical_conversion(col).show()

    segment_result.plot_closed_lost_reasons().show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Duration Analysis
    """)
    return


@app.cell
def _(result):
    result.duration_features
    for duration_feature in result.duration_features:
        result.plot_duration_distribution(duration_feature).show()
    return


if __name__ == "__main__":
    app.run()
