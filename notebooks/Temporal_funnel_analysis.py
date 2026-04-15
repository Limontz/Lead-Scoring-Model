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
    import polars as pl
    from dataclasses import dataclass
    import polars as pl
    import plotly.express as px
    import plotly.graph_objects as go

    return dataclass, go, mo, pl, px


@app.cell
def _():
    FUNNEL_STAGE_DATE_COLS = {
        "created": "DEAL_CREATEDATE",
        "mql": "DEAL_MQL_DATETIME",
        "sql": "DEAL_SQL_DATETIME",
        "opportunity": "DEAL_OPPORTUNITY_DATETIME",
        "closed_won": "DEAL_CLOSED_WON_DATE",
        "closed_lost": "DEAL_DATETIME_ENTERED_CLOSEDLOST",
    }

    MONTHLY_CONVERSION_COLS = [
        "conv_created_to_mql",
        "conv_mql_to_sql",
        "conv_sql_to_opp",
        "conv_opp_to_won",
    ]

    MONTHLY_DURATION_COLS = [
        "creation_to_mql_days",
        "mql_to_sql_days",
        "sql_to_opp_days",
        "opp_to_won_days",
        "opp_to_lost_days",
        "created_to_mql_days",
        "created_to_sql_days",
        "created_to_opp_days",
        "created_to_won_days",
        "created_to_lost_days",
    ]
    return FUNNEL_STAGE_DATE_COLS, MONTHLY_DURATION_COLS


@app.cell
def _(dataclass, pl):
    @dataclass
    class TemporalFunnelAnalysisResults:
        prepared_df: pl.DataFrame
        monthly_stage_entries: pl.DataFrame
        monthly_created_cohort: pl.DataFrame
        monthly_speed: pl.DataFrame
        segment_summary: pl.DataFrame
        monthly_segment_trend: pl.DataFrame

    return (TemporalFunnelAnalysisResults,)


@app.cell
def _(read_lead_data):
    lead_df = read_lead_data()
    return (lead_df,)


@app.cell
def _(
    lead_df,
    plot_monthly_conversion_trends,
    plot_monthly_segment_trend,
    plot_monthly_speed_trends,
    plot_monthly_stage_entries,
    plot_segment_created_to_won,
    run_temporal_funnel_analysis,
):
    segment = "LEAD_TYPE"

    results = run_temporal_funnel_analysis(
        lead_df,
        segment_col=segment,
        min_segment_count=0,
        min_segment_month_count=0,
    )

    fig1 = plot_monthly_stage_entries(results.monthly_stage_entries)
    fig1.show()

    fig2 = plot_monthly_conversion_trends(results.monthly_created_cohort)
    fig2.show()

    fig3 = plot_monthly_speed_trends(results.monthly_speed)
    fig3.show()

    fig4 = plot_segment_created_to_won(
        results.segment_summary,
        segment_col=segment,
    )
    fig4.show()

    fig5 = plot_monthly_segment_trend(
        results.monthly_segment_trend,
        segment_col=segment,
    )
    fig5.show()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Helper Functions
    """)
    return


@app.cell
def _(pl):
    def read_lead_data() -> pl.DataFrame:
        df = pl.read_csv("data/lead_data.csv")

        datetime_cols = [
            "DEAL_CREATEDATE",
            "DEAL_MQL_DATETIME",
            "DEAL_SQL_DATETIME",
            "DEAL_OPPORTUNITY_DATETIME",
            "DEAL_CLOSED_WON_DATE",
            "DEAL_DATETIME_ENTERED_CLOSEDLOST",
        ]

        converted_df = df.with_columns(
            [
                pl.col(date_col).str.to_datetime("%Y-%m-%d %H:%M:%S")
                for date_col in datetime_cols
            ]
        )

        return converted_df

    return (read_lead_data,)


@app.cell
def _(FUNNEL_STAGE_DATE_COLS, pl):
    def add_month_columns(
        df: pl.DataFrame,
        stage_date_cols: dict[str, str] = FUNNEL_STAGE_DATE_COLS,
    ) -> pl.DataFrame:

        exprs = []
        for stage_name, date_col in stage_date_cols.items():
            exprs.append(
                pl.col(date_col).dt.truncate("1mo").alias(f"{stage_name}_month")
            )
        return df.with_columns(exprs)

    def compute_stage_durations(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            [
                (pl.col("DEAL_MQL_DATETIME") - pl.col("DEAL_CREATEDATE"))
                .dt.total_days()
                .alias("creation_to_mql_days"),
                (pl.col("DEAL_SQL_DATETIME") - pl.col("DEAL_MQL_DATETIME"))
                .dt.total_days()
                .alias("mql_to_sql_days"),
                (pl.col("DEAL_OPPORTUNITY_DATETIME") - pl.col("DEAL_SQL_DATETIME"))
                .dt.total_days()
                .alias("sql_to_opp_days"),
                (pl.col("DEAL_CLOSED_WON_DATE") - pl.col("DEAL_OPPORTUNITY_DATETIME"))
                .dt.total_days()
                .alias("opp_to_won_days"),
                (
                    pl.col("DEAL_DATETIME_ENTERED_CLOSEDLOST")
                    - pl.col("DEAL_OPPORTUNITY_DATETIME")
                )
                .dt.total_days()
                .alias("opp_to_lost_days"),
                (pl.col("DEAL_MQL_DATETIME") - pl.col("DEAL_CREATEDATE"))
                .dt.total_days()
                .alias("created_to_mql_days"),
                (pl.col("DEAL_SQL_DATETIME") - pl.col("DEAL_CREATEDATE"))
                .dt.total_days()
                .alias("created_to_sql_days"),
                (pl.col("DEAL_OPPORTUNITY_DATETIME") - pl.col("DEAL_CREATEDATE"))
                .dt.total_days()
                .alias("created_to_opp_days"),
                (pl.col("DEAL_CLOSED_WON_DATE") - pl.col("DEAL_CREATEDATE"))
                .dt.total_days()
                .alias("created_to_won_days"),
                (pl.col("DEAL_DATETIME_ENTERED_CLOSEDLOST") - pl.col("DEAL_CREATEDATE"))
                .dt.total_days()
                .alias("created_to_lost_days"),
            ]
        )

    return add_month_columns, compute_stage_durations


@app.cell
def _(FUNNEL_STAGE_DATE_COLS, MONTHLY_DURATION_COLS, pl):
    def add_stage_flags(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            [
                pl.col("DEAL_MQL_DATETIME").is_not_null().alias("is_mql"),
                pl.col("DEAL_SQL_DATETIME").is_not_null().alias("is_sql"),
                pl.col("DEAL_OPPORTUNITY_DATETIME")
                .is_not_null()
                .alias("is_opportunity"),
                pl.col("DEAL_CLOSED_WON_DATE").is_not_null().alias("is_closed_won"),
                pl.col("DEAL_DATETIME_ENTERED_CLOSEDLOST")
                .is_not_null()
                .alias("is_closed_lost"),
            ]
        )

    def add_conversion_flags(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            [
                pl.col("is_mql").alias("conv_created_to_mql"),
                pl.col("is_sql").alias("conv_mql_to_sql"),
                pl.col("is_opportunity").alias("conv_sql_to_opp"),
                pl.col("is_closed_won").alias("conv_opp_to_won"),
            ]
        )

    def compute_monthly_stage_entries(
        df: pl.DataFrame,
        stage_date_cols: dict[str, str] = FUNNEL_STAGE_DATE_COLS,
    ) -> pl.DataFrame:
        """
        counts how many deals entered each stage in each month.
        """
        frames = []

        for stage_name in stage_date_cols:
            month_col = f"{stage_name}_month"

            stage_df = (
                df.filter(pl.col(month_col).is_not_null())
                .group_by(month_col)
                .agg(pl.col("DEAL_ID").n_unique().alias("n_deals"))
                .rename({month_col: "month"})
                .with_columns(pl.lit(stage_name).alias("stage"))
                .select(["month", "stage", "n_deals"])
            )
            frames.append(stage_df)

        return pl.concat(frames).sort(["month", "stage"])

    def compute_monthly_created_cohort_metrics(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.filter(pl.col("created_month").is_not_null())
            .group_by("created_month")
            .agg(
                [
                    pl.col("DEAL_ID").n_unique().alias("created"),
                    pl.col("is_mql").sum().alias("mql"),
                    pl.col("is_sql").sum().alias("sql"),
                    pl.col("is_opportunity").sum().alias("opportunity"),
                    pl.col("is_closed_won").sum().alias("closed_won"),
                    pl.col("is_closed_lost").sum().alias("closed_lost"),
                ]
            )
            .with_columns(
                [
                    (pl.col("mql") / pl.col("created")).alias("cr_created_to_mql"),
                    (pl.col("sql") / pl.col("mql")).alias("cr_mql_to_sql"),
                    (pl.col("opportunity") / pl.col("sql")).alias("cr_sql_to_opp"),
                    (pl.col("closed_won") / pl.col("opportunity")).alias(
                        "cr_opp_to_won"
                    ),
                    (pl.col("closed_won") / pl.col("created")).alias(
                        "cr_created_to_won"
                    ),
                    (pl.col("closed_lost") / pl.col("created")).alias(
                        "cr_created_to_lost"
                    ),
                ]
            )
            .sort("created_month")
        )

    def compute_monthly_speed_metrics(
        df: pl.DataFrame,
        duration_cols: list[str] | None = None,
    ) -> pl.DataFrame:
        """
        Monthly average/median durations by creation month.
        """
        duration_cols = duration_cols or MONTHLY_DURATION_COLS

        agg_exprs = []
        for col_name in duration_cols:
            agg_exprs.extend(
                [
                    pl.col(col_name).mean().alias(f"{col_name}_mean"),
                ]
            )

        return df.group_by("created_month").agg(agg_exprs).sort("created_month")

    def compute_monthly_segment_trend(
        df: pl.DataFrame,
        segment_col: str = "DEAL_DEALSOURCE",
        min_count_per_month: int = 0,
    ) -> pl.DataFrame:

        return (
            df.group_by(["created_month", segment_col])
            .agg(
                [
                    pl.col("DEAL_ID").n_unique().alias("created"),
                    pl.col("is_closed_won").sum().alias("closed_won"),
                    pl.col("is_closed_lost").sum().alias("closed_lost"),
                ]
            )
            .filter(pl.col("created") >= min_count_per_month)
            .with_columns(
                [
                    (pl.col("closed_won") / pl.col("created")).alias(
                        "cr_created_to_won"
                    ),
                    (pl.col("closed_lost") / pl.col("created")).alias(
                        "cr_created_to_lost"
                    ),
                ]
            )
            .sort(["created_month", segment_col])
        )

    return (
        add_conversion_flags,
        add_stage_flags,
        compute_monthly_created_cohort_metrics,
        compute_monthly_segment_trend,
        compute_monthly_speed_metrics,
        compute_monthly_stage_entries,
    )


@app.cell
def _(
    TemporalFunnelAnalysisResults,
    add_conversion_flags,
    add_month_columns,
    add_stage_flags,
    compute_monthly_created_cohort_metrics,
    compute_monthly_segment_trend,
    compute_monthly_speed_metrics,
    compute_monthly_stage_entries,
    compute_stage_durations,
    pl,
):
    def prepare_funnel_analysis_df(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.pipe(add_stage_flags)
            .pipe(add_conversion_flags)
            .pipe(compute_stage_durations)
        )

    def prepare_temporal_analysis_df(df: pl.DataFrame) -> pl.DataFrame:
        return df.pipe(prepare_funnel_analysis_df).pipe(add_month_columns)

    def compute_segment_funnel_summary(
        df: pl.DataFrame,
        segment_col: str = "DEAL_DEALSOURCE",
        min_count: int = 0,
    ) -> pl.DataFrame:
        return (
            df.group_by(segment_col)
            .agg(
                [
                    pl.col("DEAL_ID").n_unique().alias("created"),
                    pl.col("is_mql").sum().alias("mql"),
                    pl.col("is_sql").sum().alias("sql"),
                    pl.col("is_opportunity").sum().alias("opportunity"),
                    pl.col("is_closed_won").sum().alias("closed_won"),
                    pl.col("is_closed_lost").sum().alias("closed_lost"),
                    pl.col("created_to_won_days")
                    .median()
                    .alias("median_created_to_won_days"),
                    pl.col("DEAL_AMOUNT").mean().alias("avg_deal_amount"),
                ]
            )
            .filter(pl.col("created") >= min_count)
            .with_columns(
                [
                    (pl.col("mql") / pl.col("created")).alias("cr_created_to_mql"),
                    (pl.col("sql") / pl.col("mql")).alias("cr_mql_to_sql"),
                    (pl.col("opportunity") / pl.col("sql")).alias("cr_sql_to_opp"),
                    (pl.col("closed_won") / pl.col("opportunity")).alias(
                        "cr_opp_to_won"
                    ),
                    (pl.col("closed_won") / pl.col("created")).alias(
                        "cr_created_to_won"
                    ),
                    (pl.col("closed_lost") / pl.col("created")).alias(
                        "cr_created_to_lost"
                    ),
                ]
            )
            .sort("created", descending=True)
        )

    def run_temporal_funnel_analysis(
        df: pl.DataFrame,
        segment_col: str = "DEAL_DEALSOURCE",
        min_segment_count: int = 30,
        min_segment_month_count: int = 10,
    ) -> TemporalFunnelAnalysisResults:

        prepared_df = prepare_temporal_analysis_df(df)

        return TemporalFunnelAnalysisResults(
            prepared_df=prepared_df,
            monthly_stage_entries=compute_monthly_stage_entries(prepared_df),
            monthly_created_cohort=compute_monthly_created_cohort_metrics(prepared_df),
            monthly_speed=compute_monthly_speed_metrics(prepared_df),
            segment_summary=compute_segment_funnel_summary(
                prepared_df,
                segment_col=segment_col,
                min_count=min_segment_count,
            ),
            monthly_segment_trend=compute_monthly_segment_trend(
                prepared_df,
                segment_col=segment_col,
                min_count_per_month=min_segment_month_count,
            ),
        )

    return (run_temporal_funnel_analysis,)


@app.cell
def _(go, pl, px):
    def plot_monthly_stage_entries(
        monthly_stage_entries_df: pl.DataFrame,
    ) -> go.Figure:

        fig = px.line(
            monthly_stage_entries_df,
            x="month",
            y="n_deals",
            color="stage",
            markers=True,
            title="Monthly funnel stage entries",
        )
        fig.update_layout(
            template="plotly_white",
            xaxis_title="Month",
            yaxis_title="Number of deals",
        )
        return fig

    def plot_monthly_conversion_trends(
        monthly_cohort_df: pl.DataFrame,
        conversion_cols: list[str] = [
            "cr_created_to_mql",
            "cr_mql_to_sql",
            "cr_sql_to_opp",
            "cr_opp_to_won",
            "cr_created_to_won",
        ],
    ) -> go.Figure:
        plot_df = monthly_cohort_df.select(["created_month"] + conversion_cols).unpivot(
            on=conversion_cols,
            index="created_month",
            variable_name="metric",
            value_name="conversion_rate",
        )

        fig = px.line(
            plot_df,
            x="created_month",
            y="conversion_rate",
            color="metric",
            markers=True,
            title="Monthly funnel conversion trends (creation cohort)",
        )
        fig.update_layout(
            template="plotly_white",
            xaxis_title="Creation month",
            yaxis_title="Conversion rate",
        )
        fig.update_yaxes(tickformat=".0%")
        return fig

    def plot_monthly_speed_trends(
        monthly_speed_df: pl.DataFrame,
        metric_suffix: str = "_mean",
    ) -> go.Figure:
        cols = [c for c in monthly_speed_df.columns if c.endswith(metric_suffix)]

        plot_df = monthly_speed_df.select(["created_month"] + cols).unpivot(
            on=cols,
            index="created_month",
            variable_name="metric",
            value_name="days",
        )

        fig = px.line(
            plot_df,
            x="created_month",
            y="days",
            color="metric",
            markers=True,
            title=f"Monthly funnel speed trends ({metric_suffix.replace('_', '').upper()})",
        )
        fig.update_layout(
            template="plotly_white",
            xaxis_title="Creation month",
            yaxis_title="Days",
        )
        return fig

    def plot_segment_created_to_won(
        segment_summary_df: pl.DataFrame,
        segment_col: str = "DEAL_DEALSOURCE",
        top_n: int | None = None,
    ) -> go.Figure:
        plot_df = segment_summary_df.sort("cr_created_to_won", descending=True)

        if top_n is not None:
            plot_df = plot_df.head(top_n)

        fig = px.bar(
            plot_df,
            x="cr_created_to_won",
            y=segment_col,
            orientation="h",
            text="created",
            hover_data=["created", "closed_won", "cr_created_to_won"],
            title=f"Created → Won conversion by {segment_col}",
        )
        fig.update_layout(
            template="plotly_white",
            yaxis={"categoryorder": "total ascending"},
            xaxis_title="Created → Won conversion rate",
            yaxis_title=segment_col,
        )
        fig.update_xaxes(tickformat=".0%")
        return fig

    def plot_monthly_segment_trend(
        monthly_segment_df: pl.DataFrame,
        segment_col: str = "DEAL_DEALSOURCE",
    ) -> go.Figure:

        fig = px.line(
            monthly_segment_df,
            x="created_month",
            y="cr_created_to_won",
            color=segment_col,
            markers=True,
            title=f"Monthly Created → Won trend by {segment_col}",
        )
        fig.update_layout(
            template="plotly_white",
            xaxis_title="Creation month",
            yaxis_title="Created → Won conversion rate",
        )
        fig.update_yaxes(tickformat=".0%")
        return fig

    return (
        plot_monthly_conversion_trends,
        plot_monthly_segment_trend,
        plot_monthly_speed_trends,
        plot_monthly_stage_entries,
        plot_segment_created_to_won,
    )


if __name__ == "__main__":
    app.run()
