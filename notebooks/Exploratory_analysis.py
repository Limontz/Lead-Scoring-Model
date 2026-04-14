import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # DATA EXPLORATION


    The data are quite clean. I found the following columns (other than the date columns) to contain none values:

    - DEAL_CCNL_MACRO
    - DEAL_SOURCE_DETAIL
    - DEAL_MODULI_AGGIUNTIVI_ACQUISTATI
    - DEAL_OWNER_ID
    - DEAL_BDR_OWNER_ID
    - DEAL_DEMO_SCORE
    - DEAL_NUMBER_TIMES_CONTACTED
    - UTM_SOURCE
    - COMPANY_REVENUE
    - COMPANY_FUNDING_YEAR

    None of them represent a big problem and will understand what to do with the nones once I will understand how to use these information.
    Only DEAL_DEMO_SCORE DEAL_NUMBER_TIMES_CONTACTED are strange. They are both none at the same time and none of the leads with none values in these columns are ever closed. However, some of them are SQL which means a demo was booked. Does it mean the client did not show up at the demo?

    There are some **inactive leads**, i.e. leads that were stored in the CRM but did not move to subsequent steps or flagged as lost.

    There are some **leads without any final state**. These leads cannot be used in the model since we do not know their final result.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import plotly.graph_objects as go
    from datetime import date

    return date, go, mo, pl


@app.cell
def _(read_lead_data):
    lead_df = read_lead_data()
    return (lead_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Null Exploration
    """)
    return


@app.cell
def _(lead_df):
    lead_df.null_count()
    return


@app.cell
def _(lead_df, pl):
    lead_df.filter(pl.col("DEAL_CCNL_MACRO").is_null()) # sembra ok
    return


@app.cell
def _(lead_df, pl):
    lead_df.filter(pl.col("DEAL_SOURCE_DETAIL").is_null()) # sembra ok
    return


@app.cell
def _(lead_df, pl):
    lead_df.filter(pl.col("DEAL_DEMO_SCORE").is_null(),
                   pl.col("DEAL_NUMBER_TIMES_CONTACTED").is_null()
                  )
    return


@app.cell
def _(lead_df, pl):
    lead_df.filter(pl.col("UTM_SOURCE").is_null())
    return


@app.cell
def _(lead_df, pl):
    lead_df.filter(pl.col("COMPANY_REVENUE").is_null())
    return


@app.cell
def _(lead_df, pl):
    lead_df.filter(pl.col("COMPANY_FUNDING_YEAR").is_null())
    return


@app.cell
def _(date, lead_df, pl):
    lead_df.filter(pl.col("DEAL_CREATEDATE") > date(2026, 4, 8))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Value Distribution
    """)
    return


@app.cell
def _(lead_df, plot_distributions):
    plot_distributions(lead_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Funnel Data Quality
    """)
    return


@app.cell
def _(
    check_closed_consistency,
    check_final_state_consistency,
    check_stage_progression,
    check_temporal_order,
    lead_df,
):
    print(check_stage_progression(lead_df))
    print(check_temporal_order(lead_df))
    print(check_closed_consistency(lead_df))
    print(check_final_state_consistency(lead_df))
    return


@app.cell
def _(check_inactive_deals, lead_df):
    inactive_deals_df = check_inactive_deals(lead_df)
    inactive_deals_df
    return


@app.cell
def _(get_leads_wo_final_state, lead_df):
    leads_wo_final_state = get_leads_wo_final_state(lead_df)
    leads_wo_final_state
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Helper Functions
    """)
    return


@app.cell
def _(pl):
    def read_lead_data():
        df = pl.read_csv("data/lead_data.csv")

        datetime_cols =[
            "DEAL_CREATEDATE",
            "DEAL_MQL_DATETIME",
            "DEAL_SQL_DATETIME",
            "DEAL_OPPORTUNITY_DATETIME",
            "DEAL_CLOSED_WON_DATE",
            "DEAL_DATETIME_ENTERED_CLOSEDLOST"
        ]

        converted_df = df.with_columns([
            pl.col(date_col).str.to_datetime("%Y-%m-%d %H:%M:%S")
            for date_col in datetime_cols
        ])

        return converted_df



    return (read_lead_data,)


@app.cell
def _(go, pl):
    def plot_distributions(df: pl.DataFrame, max_categories: int = 9999999):
        non_dt_cols = [
            col for col in df.columns
            if not isinstance(df[col].dtype, (pl.Datetime, pl.Date, pl.Duration))
        ]
        for col_name in non_dt_cols:
            series = df[col_name]
            dtype = df[col_name].dtype

            if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                         pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                         pl.Float32, pl.Float64):

                col_min = series.drop_nulls().min()
                col_max = series.drop_nulls().max()
                print(f"{col_name} → min: {col_min}, max: {col_max}")

                series_filled = series.fill_null(-9999).fill_nan(-9999)
                fig = go.Figure(
                    go.Histogram(
                        x=series_filled.to_list(),
                        marker_color="steelblue",
                        opacity=0.8,
                    )
                )
            else:
                series_filled = series.fill_null("NULL")
                counts = (
                    series_filled
                    .value_counts()
                    .sort("count", descending=True)
                    .head(max_categories)
                )
                fig = go.Figure(
                    go.Bar(
                        x=counts[col_name].to_list(),
                        y=counts["count"].to_list(),
                        marker_color="coral",
                        opacity=0.8,
                    )
                )

            fig.update_layout(
                title_text=col_name,
                height=400,
                bargap=0.1,
            )
            fig.show()

    return (plot_distributions,)


@app.cell
def _(pl):
    def check_stage_progression(df: pl.DataFrame):
        return {
            "sql_without_mql": df.filter(
                pl.col("DEAL_SQL_DATETIME").is_not_null() &
                pl.col("DEAL_MQL_DATETIME").is_null()
            ).select(pl.col("DEAL_ID").n_unique()).item(),

            "opportunity_without_sql": df.filter(
                pl.col("DEAL_OPPORTUNITY_DATETIME").is_not_null() &
                pl.col("DEAL_SQL_DATETIME").is_null()
            ).select(pl.col("DEAL_ID").n_unique()).item(),

            "won_without_opportunity": df.filter(
                pl.col("DEAL_CLOSED_WON_DATE").is_not_null() &
                pl.col("DEAL_OPPORTUNITY_DATETIME").is_null()
            ).select(pl.col("DEAL_ID").n_unique()).item(),
        }

    def check_temporal_order(df: pl.DataFrame):
        return {
            "mql_before_created": df.filter(
                pl.col("DEAL_MQL_DATETIME") < pl.col("DEAL_CREATEDATE")
            ).select(pl.col("DEAL_ID").n_unique()).item(),

            "sql_before_mql": df.filter(
                pl.col("DEAL_SQL_DATETIME") < pl.col("DEAL_MQL_DATETIME")
            ).select(pl.col("DEAL_ID").n_unique()).item(),

            "opp_before_sql": df.filter(
                pl.col("DEAL_OPPORTUNITY_DATETIME") < pl.col("DEAL_SQL_DATETIME")
            ).select(pl.col("DEAL_ID").n_unique()).item(),

            "won_before_opp": df.filter(
                pl.col("DEAL_CLOSED_WON_DATE") < pl.col("DEAL_OPPORTUNITY_DATETIME")
            ).select(pl.col("DEAL_ID").n_unique()).item(),
        }

    def check_closed_consistency(df: pl.DataFrame):
        return {
            "conflicting_deals": df.filter(
                pl.col("DEAL_CLOSED_WON_DATE").is_not_null() &
                pl.col("DEAL_DATETIME_ENTERED_CLOSEDLOST").is_not_null()
            ).select(pl.col("DEAL_ID").n_unique()).item()
        }

    def check_inactive_deals(df: pl.DataFrame):
        return {
            "inactive_deals": df.filter(
                pl.col("DEAL_MQL_DATETIME").is_null() &
                pl.col("DEAL_SQL_DATETIME").is_null() &
                pl.col("DEAL_OPPORTUNITY_DATETIME").is_null() &
                pl.col("DEAL_CLOSED_WON_DATE").is_null() &
                pl.col("DEAL_DATETIME_ENTERED_CLOSEDLOST").is_null()
            )
        }

    def check_final_state_consistency(df: pl.DataFrame):
        both_closed = df.filter(
            pl.col("DEAL_CLOSED_WON_DATE").is_not_null() &
            pl.col("DEAL_DATETIME_ENTERED_CLOSEDLOST").is_not_null()
        ).select(pl.col("DEAL_ID").n_unique()).item()

        closed_date = pl.coalesce([
            pl.col("DEAL_CLOSED_WON_DATE"),
            pl.col("DEAL_DATETIME_ENTERED_CLOSEDLOST")
        ])

        progressed_after_closed = df.filter(
            closed_date.is_not_null() & (
                (pl.col("DEAL_MQL_DATETIME") > closed_date) |
                (pl.col("DEAL_SQL_DATETIME") > closed_date) |
                (pl.col("DEAL_OPPORTUNITY_DATETIME") > closed_date)
            )
        ).select(pl.col("DEAL_ID").n_unique()).item()

        no_final_state = df.filter(
            pl.col("DEAL_CLOSED_WON_DATE").is_null() &
            pl.col("DEAL_DATETIME_ENTERED_CLOSEDLOST").is_null()
        ).select(pl.col("DEAL_ID").n_unique()).item()

        return {
            "both_won_and_lost": both_closed,
            "progressed_after_closed": progressed_after_closed,
            "no_final_state": no_final_state
        }

    def get_leads_wo_final_state(df: pl.DataFrame):

        return df.filter(
            pl.col("DEAL_CLOSED_WON_DATE").is_null() &
            pl.col("DEAL_DATETIME_ENTERED_CLOSEDLOST").is_null()
        )

    return (
        check_closed_consistency,
        check_final_state_consistency,
        check_inactive_deals,
        check_stage_progression,
        check_temporal_order,
        get_leads_wo_final_state,
    )


if __name__ == "__main__":
    app.run()
