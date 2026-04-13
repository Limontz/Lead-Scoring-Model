import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Volumes and Conversion Rate Analysis

    Il tempo tipico tra uno stage e l'altro varia tra 0 e 30 gionrni. La durata aumenta fino a 90 giorni per l'ultimo step da opprtunity a deal chiuso/perso.

    L'intero processo di acquisizione di un cliente, dalla data creazione della lead alla firma/rinuncia del contratto varia da 0 a 180 giorni con una media di circa 90 giorni.

    L'unico momento in cui si nota chiaramente che alcune variabli potrebbero essere predittive è nella fase tra opportunity e won. In questa fase si nota una chiara differenza di success rate a seconda del valore di queste variabili:

    - DEAL_DEMO_SCORE
    - DEAL_NUMBER_TIMES_CONTACTED
    - DEAL_DEALSOURCE
    - DEAL_INDUSTRY
    - CONTACT_ROLE
    - COMPANY_STATE
    """)
    return


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import plotly.graph_objects as go
    import plotly.express as px
    import numpy as np

    return go, mo, pl, px


@app.cell
def _(read_lead_data):
    lead_df = read_lead_data()
    return (lead_df,)


@app.cell
def _(compute_stage_durations, lead_df):
    stage_duration_df = compute_stage_durations(lead_df)
    return (stage_duration_df,)


@app.cell
def _(plot_stage_durations, stage_duration_df):
    plot_stage_durations(stage_duration_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Volumes and conversion rate
    """)
    return


@app.cell
def _(compute_funnel_metrics, get_sql_to_demo_score_conversion_rate, lead_df):
    metrics = compute_funnel_metrics(lead_df)

    print("### Volumes")
    for stage, vol in metrics["volumes"].items():
        print(f"- {stage}: {vol}")

    print("\n### Step-to-step conversion rates")
    for transition, rate in metrics["step_conversion_rates"].items():
        print(f"- {transition}: {rate}%")

    print("\n### Conversion rates vs created")
    for transition, rate in metrics["vs_created_conversion_rates"].items():
        print(f"- {transition}: {rate}%")

    print("\n### From SQL to demo done")
    print("rate", f"{get_sql_to_demo_score_conversion_rate(lead_df)}%")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Reasons for dropping
    """)
    return


@app.cell
def _(
    CATEGORICAL_FEATURES,
    DURATION_FEATURES,
    NUMERIC_FEATURES,
    lead_df,
    prepare_funnel_analysis_df,
    run_step_analysis,
):
    run_step_analysis(
        prepare_funnel_analysis_df(lead_df),
        "opp_to_won",
        NUMERIC_FEATURES + DURATION_FEATURES,
        CATEGORICAL_FEATURES,
        numeric_plot_kind = "box",
    )
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
def _(pl):
    def compute_stage_durations(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            (pl.col("DEAL_MQL_DATETIME") - pl.col("DEAL_CREATEDATE")).dt.total_days().alias("creation_to_mql_days"),
            (pl.col("DEAL_SQL_DATETIME") - pl.col("DEAL_MQL_DATETIME")).dt.total_days().alias("mql_to_sql_days"),
            (pl.col("DEAL_OPPORTUNITY_DATETIME") - pl.col("DEAL_SQL_DATETIME")).dt.total_days().alias("sql_to_opp_days"),
            (pl.col("DEAL_CLOSED_WON_DATE") - pl.col("DEAL_OPPORTUNITY_DATETIME")).dt.total_days().alias("opp_to_won_days"),
            (pl.col("DEAL_DATETIME_ENTERED_CLOSEDLOST") - pl.col("DEAL_OPPORTUNITY_DATETIME")).dt.total_days().alias("opp_to_lost_days"),

            (pl.col("DEAL_MQL_DATETIME") - pl.col("DEAL_CREATEDATE")).dt.total_days().alias("created_to_mql_days"),
            (pl.col("DEAL_SQL_DATETIME") - pl.col("DEAL_CREATEDATE")).dt.total_days().alias("created_to_sql_days"),
            (pl.col("DEAL_OPPORTUNITY_DATETIME") - pl.col("DEAL_CREATEDATE")).dt.total_days().alias("created_to_opp_days"),
            (pl.col("DEAL_CLOSED_WON_DATE") - pl.col("DEAL_CREATEDATE")).dt.total_days().alias("created_to_won_days"),
            (pl.col("DEAL_DATETIME_ENTERED_CLOSEDLOST") - pl.col("DEAL_CREATEDATE")).dt.total_days().alias("created_to_lost_days"),
        ])

    def compute_funnel_volumes(df: pl.DataFrame) -> dict[str, float]:
        return {
            "created": df.select(pl.col("DEAL_ID").n_unique()).item(),
            "mql": df.filter(pl.col("DEAL_MQL_DATETIME").is_not_null())
                     .select(pl.col("DEAL_ID").n_unique()).item(),
            "sql": df.filter(pl.col("DEAL_SQL_DATETIME").is_not_null())
                     .select(pl.col("DEAL_ID").n_unique()).item(),
            "opportunity": df.filter(pl.col("DEAL_OPPORTUNITY_DATETIME").is_not_null())
                             .select(pl.col("DEAL_ID").n_unique()).item(),
            "closed_won": df.filter(pl.col("DEAL_CLOSED_WON_DATE").is_not_null())
                            .select(pl.col("DEAL_ID").n_unique()).item(),
        }

    def compute_funnel_metrics(df: pl.DataFrame) -> dict[str, dict[str, float]]:
        volumes = compute_funnel_volumes(df)
        stages = ["created", "mql", "sql", "opportunity", "closed_won"]

        step_rates = {}
        for i in range(1, len(stages)):
            current_stage = stages[i]
            previous_stage = stages[i - 1]
            rate = volumes[current_stage] / volumes[previous_stage] * 100
            step_rates[f"{previous_stage}_to_{current_stage}"] = round(rate, 2)

        vs_created_rates = {}
        for stage in stages[1:]:
            rate = volumes[stage] / volumes["created"] * 100
            vs_created_rates[f"created_to_{stage}"] = round(rate, 2)

        return {
            "volumes": volumes,
            "step_conversion_rates": step_rates,
            "vs_created_conversion_rates": vs_created_rates,
        }

    def get_sql_to_demo_score_conversion_rate(df: pl.DataFrame) -> float:
        volumes = compute_funnel_volumes(df)

        lead_with_demo_score = df.filter(
            pl.col("DEAL_DEMO_SCORE").is_not_null(),
            pl.col("DEAL_SQL_DATETIME").is_not_null()
        ).select(pl.col("DEAL_ID").n_unique()).item()
        rate = lead_with_demo_score / volumes["created"] * 100

        return round(rate, 2)

    return (
        compute_funnel_metrics,
        compute_stage_durations,
        get_sql_to_demo_score_conversion_rate,
    )


@app.cell
def _(go, pl):
    def plot_stage_durations(stage_duration_df: pl.DataFrame) -> None:
        duration_cols = [col for col in stage_duration_df.columns if col.endswith("_days")]

        for col_name in duration_cols:
            series = stage_duration_df[col_name].drop_nulls()
            col_min = series.min()
            col_max = series.max()
            col_mean = series.mean()
            print(f"{col_name} → min: {col_min} days, max: {col_max} days, mean: {round(col_mean, 0)} days")

            fig = go.Figure(
                go.Histogram(
                    x=series.to_list(),
                    marker_color="steelblue",
                    opacity=0.8,
                )
            )
            fig.update_layout(
                title_text=col_name,
                xaxis_title="days",
                yaxis_title="count",
                height=400,
                bargap=0.1,
            )
            fig.show()

    return (plot_stage_durations,)


@app.cell
def _(compute_stage_durations, go, pl, px):



    STEPS = {
        "created_to_mql": {
            "current_filter": None,
            "target_col": "conv_created_to_mql",
            "title": "Created → MQL",
        },
        "mql_to_sql": {
            "current_filter": pl.col("is_mql"),
            "target_col": "conv_mql_to_sql",
            "title": "MQL → SQL",
        },
        "sql_to_opp": {
            "current_filter": pl.col("is_sql"),
            "target_col": "conv_sql_to_opp",
            "title": "SQL → Opportunity",
        },
        "opp_to_won": {
            "current_filter": pl.col("is_opportunity"),
            "target_col": "conv_opp_to_won",
            "title": "Opportunity → Closed Won",
        },
    }

    NUMERIC_FEATURES = [
        "DEAL_DEMO_SCORE",
        "DEAL_AMOUNT",
        "DEAL_NUMBER_OF_EMPLOYEES",
        "DEAL_NUMERO_CEDOLINI",
        "DEAL_NUMBER_TIMES_CONTACTED",
        "COMPANY_REVENUE",
        "COMPANY_FUNDING_YEAR",
    ]

    CATEGORICAL_FEATURES = [
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

    DURATION_FEATURES = [
        "creation_to_mql_days",
        "mql_to_sql_days",
        "sql_to_opp_days",
        "opp_to_won_days",
        "opp_to_lost_days",
    ]


    def add_stage_flags(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            pl.col("DEAL_MQL_DATETIME").is_not_null().alias("is_mql"),
            pl.col("DEAL_SQL_DATETIME").is_not_null().alias("is_sql"),
            pl.col("DEAL_OPPORTUNITY_DATETIME").is_not_null().alias("is_opportunity"),
            pl.col("DEAL_CLOSED_WON_DATE").is_not_null().alias("is_closed_won"),
            pl.col("DEAL_DATETIME_ENTERED_CLOSEDLOST").is_not_null().alias("is_closed_lost"),
        ])


    def add_conversion_flags(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns([
            pl.col("is_mql").alias("conv_created_to_mql"),
            pl.col("is_sql").alias("conv_mql_to_sql"),
            pl.col("is_opportunity").alias("conv_sql_to_opp"),
            pl.col("is_closed_won").alias("conv_opp_to_won"),
        ])


    def prepare_funnel_analysis_df(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.pipe(add_stage_flags)
              .pipe(add_conversion_flags)
              .pipe(compute_stage_durations)
        )


    def get_step_dataset(df: pl.DataFrame, step_name: str) -> pl.DataFrame:
        config = STEPS[step_name]
        if config["current_filter"] is None:
            return df
        return df.filter(config["current_filter"])

    def summarize_numeric_by_target(df: pl.DataFrame, feature: str, target_col: str) -> pl.DataFrame:
        return (
            df.filter(pl.col(feature).is_not_null())
            .group_by(target_col)
            .agg([
                pl.len().alias("n"),
                pl.col(feature).mean().alias("mean"),
                pl.col(feature).median().alias("median"),
                pl.col(feature).std().alias("std"),
                pl.col(feature).min().alias("min"),
                pl.col(feature).max().alias("max"),
            ])
            .sort(target_col)
        )


    def summarize_categorical_conversion(
        df: pl.DataFrame,
        feature: str,
        target_col: str,
        min_count: int = 30,
    ) -> pl.DataFrame:
        return (
            df.filter(pl.col(feature).is_not_null())
            .group_by(feature)
            .agg([
                pl.len().alias("n"),
                pl.col(target_col).mean().alias("conversion_rate"),
            ])
            .sort("conversion_rate", descending=True)
        )


    def plot_numeric_distribution(
        df: pl.DataFrame,
        feature: str,
        target_col: str,
        step_title: str,
        plot_kind: str = "box",  # "box", "violin", "hist"
        clip_outliers: bool = True,
    ) -> go.Figure:
        pdf = (
            df.select([feature, target_col])
            .filter(pl.col(feature).is_not_null())
        )

        if plot_kind == "box":
            fig = px.box(
                pdf,
                x=target_col,
                y=feature,
                color=target_col,
                points="outliers",
                title=f"{step_title} | {feature}",
            )
        elif plot_kind == "violin":
            fig = px.violin(
                pdf,
                x=target_col,
                y=feature,
                color=target_col,
                box=True,
                points="outliers",
                title=f"{step_title} | {feature}",
            )
        elif plot_kind == "hist":
            fig = px.histogram(
                pdf,
                x=feature,
                color=target_col,
                barmode="overlay",
                marginal="box",
                title=f"{step_title} | {feature}",
                histnorm="probability density",
            )
        else:
            raise ValueError(f"Unsupported plot_kind: {plot_kind}")

        fig.update_layout(template="plotly_white")
        fig.show()
        return fig


    def plot_categorical_conversion(
        df: pl.DataFrame,
        feature: str,
        target_col: str,
        step_title: str,
        min_count: int = 0,
    ) -> go.Figure:
        agg = summarize_categorical_conversion(
            df=df,
            feature=feature,
            target_col=target_col,
            min_count=min_count,
        )

        fig = px.bar(
            agg,
            x="conversion_rate",
            y=feature,
            orientation="h",
            text="n",
            hover_data=["n", "conversion_rate"],
            title=f"{step_title} | Conversion rate by {feature}",
        )

        fig.update_layout(
            template="plotly_white",
            xaxis_tickformat=".0%",
            yaxis={"categoryorder": "total ascending"},
        )
        fig.show()
        return fig

    def plot_closed_lost_reasons(
        df: pl.DataFrame,
        step_title: str,
        top_n: int = 15,
    ) -> None:
        plot_df = (
            df.filter(pl.col("is_closed_lost"))
            .filter(pl.col("DEAL_CLOSED_LOST_REASON").is_not_null())
            .group_by("DEAL_CLOSED_LOST_REASON")
            .agg(pl.len().alias("n"))
            .sort("n", descending=True)
            .head(top_n)
        )

        if plot_df.is_empty():
            print("No closed lost reasons available")
            return

        fig = px.bar(
            plot_df,
            x="n",
            y="DEAL_CLOSED_LOST_REASON",
            orientation="h",
            text="n",
            title=f"{step_title} | Closed Lost Reasons",
        )

        fig.update_layout(
            template="plotly_white",
            yaxis={"categoryorder": "total ascending"},
        )

        fig.show()


    def run_step_analysis(
        df: pl.DataFrame,
        step_name: str,
        numeric_features: list[str],
        categorical_features: list[str],
        numeric_plot_kind: str = "box",
    ) -> dict[str, dict[str, pl.DataFrame]]:
        config = STEPS[step_name]
        step_df = get_step_dataset(df, step_name)
        target_col = config["target_col"]
        step_title = config["title"]

        results = {
            "numeric_summaries": {},
            "categorical_summaries": {},
        }

        conversion_rate = step_df.select(pl.col(target_col).mean()).item()

        print(f"\n{'=' * 80}")
        print(step_title)
        print(f"Rows in scope: {step_df.height}")
        print(f"Conversion rate: {conversion_rate:.2%}")
        print(f"{'=' * 80}\n")

        for feature in numeric_features:
            summary = summarize_numeric_by_target(step_df, feature, target_col)
            results["numeric_summaries"][feature] = summary

            print(f"\n[NUMERIC] {feature}")
            #print(summary)

            plot_numeric_distribution(
                df=step_df,
                feature=feature,
                target_col=target_col,
                step_title=step_title,
                plot_kind=numeric_plot_kind,
            )

        for feature in categorical_features:
            summary = summarize_categorical_conversion(
                df=step_df,
                feature=feature,
                target_col=target_col,
                min_count=30,
            )
            results["categorical_summaries"][feature] = summary

            print(f"\n[CATEGORICAL] {feature}")
            #print(summary.head(15))

            plot_categorical_conversion(
                df=step_df,
                feature=feature,
                target_col=target_col,
                step_title=step_title,
                min_count=30,
            )

        plot_closed_lost_reasons(
            step_df,
            step_title=step_title,
        )

        return results


    def run_full_funnel_feature_analysis(
        df: pl.DataFrame,
        numeric_plot_kind: str = "box",
    ) -> dict[str, dict[str, dict[str, pl.DataFrame]]]:
        df = prepare_funnel_analysis_df(df)
        all_numeric = NUMERIC_FEATURES + DURATION_FEATURES

        results = {}
        for step_name in STEPS:
            results[step_name] = run_step_analysis(
                df=df,
                step_name=step_name,
                numeric_features=all_numeric,
                categorical_features=CATEGORICAL_FEATURES,
                numeric_plot_kind=numeric_plot_kind,
            )

        return results

    return (
        CATEGORICAL_FEATURES,
        DURATION_FEATURES,
        NUMERIC_FEATURES,
        prepare_funnel_analysis_df,
        run_step_analysis,
    )


if __name__ == "__main__":
    app.run()
