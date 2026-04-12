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
    """)
    return


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import plotly.graph_objects as go

    return go, mo, pl


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


if __name__ == "__main__":
    app.run()
