import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Funnel Time Series Analysis

    #### Results:

    - number of deals increased with time
    - conversion rate between stages is constant over time
    - velocity of stages is constant over time
    """)
    return


@app.cell
def _():
    import marimo as mo
    import json
    from dataclasses import asdict

    from lead_scoring.analysis.funnel_analysis import analyze_temporal
    from lead_scoring.data.io import read_lead_data
    from lead_scoring.data.manipulations import prepare_data_for_analysis
    from lead_scoring.data.cleaning import (
        cast_datetime_columns,
        drop_future_terminal_deals,
        drop_disqualified_deals,
    )
    from lead_scoring.data.validation import build_validation_report

    return (
        analyze_temporal,
        asdict,
        build_validation_report,
        cast_datetime_columns,
        drop_future_terminal_deals,
        json,
        mo,
        prepare_data_for_analysis,
        read_lead_data,
    )


@app.cell
def _(
    analyze_temporal,
    asdict,
    build_validation_report,
    cast_datetime_columns,
    drop_future_terminal_deals,
    json,
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
    df = drop_future_terminal_deals(df)

    enriched_df = prepare_data_for_analysis(df)
    print("Data enriched successfully.")

    result = analyze_temporal(enriched_df)

    result.summary()
    result.available_features()
    result.plot_stage_entries().show()
    result.plot_conversion_trends().show()
    result.plot_duration_trends().show()
    result.plot_sql_to_demo_score_trends().show()
    return


if __name__ == "__main__":
    app.run()
