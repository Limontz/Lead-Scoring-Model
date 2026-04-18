import polars as pl
import plotly.express as px
import plotly.graph_objects as go


def plot_stage_volumes(
    volumes: dict[str, int],
    title: str = "Funnel volumes",
) -> go.Figure:
    plot_df = pl.DataFrame(
        {
            "stage": list(volumes.keys()),
            "n_deals": list(volumes.values()),
        }
    )

    fig = px.bar(
        plot_df,
        x="stage",
        y="n_deals",
        text="n_deals",
        title=title,
    )
    fig.update_layout(template="plotly_white", xaxis_title="Stage", yaxis_title="Deals")
    return fig


def plot_segment_stage_volumes(
    plot_df: pl.DataFrame,
    segment_col: str,
    title: str = "Funnel volumes by segment",
) -> go.Figure:
    fig = px.bar(
        plot_df,
        x="stage",
        y="n_deals",
        color=segment_col,
        barmode="group",
        title=title,
    )
    fig.update_layout(template="plotly_white", xaxis_title="Stage", yaxis_title="Deals")
    return fig


def plot_conversion_rates(
    rates: dict[str, float | None],
    title: str = "Conversion rates",
) -> go.Figure:
    plot_df = pl.DataFrame(
        {
            "transition": list(rates.keys()),
            "conversion_rate": list(rates.values()),
        }
    )

    fig = px.bar(
        plot_df,
        x="transition",
        y="conversion_rate",
        text="conversion_rate",
        title=title,
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Transition",
        yaxis_title="Conversion rate",
    )
    fig.update_yaxes(tickformat=".0%")
    return fig


def plot_segment_conversion_rates(
    plot_df: pl.DataFrame,
    segment_col: str,
    title: str = "Conversion rates by segment",
) -> go.Figure:
    fig = px.bar(
        plot_df,
        x="transition",
        y="conversion_rate",
        color=segment_col,
        barmode="group",
        title=title,
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Transition",
        yaxis_title="Conversion rate",
    )
    fig.update_yaxes(tickformat=".0%")
    return fig


def plot_numeric_boxplot(
    df: pl.DataFrame,
    feature: str,
    target_col: str,
    title: str,
    segment_col: str | None = None,
) -> go.Figure:
    cols = [feature, target_col] + ([segment_col] if segment_col else [])
    plot_df = df.select(cols).filter(pl.col(feature).is_not_null())

    fig = px.box(
        plot_df,
        x=target_col,
        y=feature,
        color=segment_col or target_col,
        points="outliers",
        title=title,
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_numeric_histogram(
    df: pl.DataFrame,
    feature: str,
    target_col: str,
    title: str,
    segment_col: str | None = None,
) -> go.Figure:
    cols = [feature, target_col] + ([segment_col] if segment_col else [])
    plot_df = df.select(cols).filter(pl.col(feature).is_not_null())

    color_col = segment_col or target_col

    fig = px.histogram(
        plot_df,
        x=feature,
        color=color_col,
        barmode="overlay",
        marginal="box",
        histnorm="probability density",
        title=title,
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_duration_distribution(
    df: pl.DataFrame,
    feature: str,
    title: str,
    segment_col: str | None = None,
) -> go.Figure:
    cols = [feature] + ([segment_col] if segment_col else [])
    plot_df = df.select(cols).filter(pl.col(feature).is_not_null())

    fig = px.histogram(
        plot_df,
        x=feature,
        color=segment_col,
        marginal="box",
        title=title,
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Days",
        yaxis_title="Count",
    )
    return fig


def plot_categorical_conversion(
    summary_df: pl.DataFrame,
    feature: str,
    title: str,
    segment_col: str | None = None,
) -> go.Figure:
    fig = px.bar(
        summary_df,
        x="conversion_rate",
        y=feature,
        color=segment_col,
        orientation="h",
        text="n",
        hover_data=["n", "conversion_rate"],
        title=title,
        barmode="group" if segment_col else "relative",
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Conversion rate",
        yaxis_title=feature,
        yaxis={"categoryorder": "total ascending"},
    )
    fig.update_xaxes(tickformat=".0%")
    return fig


def plot_closed_lost_reasons(
    reason_df: pl.DataFrame,
    reason_col: str = "DEAL_CLOSED_LOST_REASON",
    title: str = "Closed lost reasons",
    segment_col: str | None = None,
) -> go.Figure:
    fig = px.bar(
        reason_df,
        x="n",
        y=reason_col,
        color=segment_col,
        orientation="h",
        text="n",
        title=title,
        barmode="group" if segment_col else "relative",
    )
    fig.update_layout(
        template="plotly_white",
        yaxis={"categoryorder": "total ascending"},
        xaxis_title="Deals",
        yaxis_title=reason_col,
    )
    return fig


def plot_stage_entries(df: pl.DataFrame) -> go.Figure:
    fig = px.line(
        df,
        x="month",
        y="n_deals",
        color="stage",
        title="Stage entries over time",
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_conversion_trends(
    df: pl.DataFrame,
    conversion_cols: list[str],
) -> go.Figure:
    plot_df = df.select(["created_month"] + conversion_cols).unpivot(
        index="created_month",
        variable_name="metric",
        value_name="value",
    )

    fig = px.line(
        plot_df,
        x="created_month",
        y="value",
        color="metric",
        title="Conversion trends",
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(template="plotly_white")
    return fig


def plot_segment_trends(
    df: pl.DataFrame,
    segment_col: str,
) -> go.Figure:
    fig = px.line(
        df,
        x="created_month",
        y="cr_created_to_won",
        color=segment_col,
        title=f"Conversion by {segment_col} over time",
    )
    fig.update_yaxes(tickformat=".0%")
    return fig


def plot_duration_trends(df: pl.DataFrame) -> go.Figure:
    plot_df = df.unpivot(
        index="created_month",
        variable_name="metric",
        value_name="value",
    )

    fig = px.line(
        plot_df,
        x="created_month",
        y="value",
        color="metric",
        title="Duration trends",
    )
    return fig


def plot_monthly_stage_entries(
    df: pl.DataFrame,
    title: str = "Monthly funnel stage entries",
    segment_col: str | None = None,
) -> go.Figure:
    color_col = segment_col if segment_col else "stage"
    line_group = "stage" if segment_col else None

    fig = px.line(
        df,
        x="month",
        y="n_deals",
        color=color_col,
        line_dash="stage" if segment_col else None,
        line_group=line_group,
        markers=True,
        title=title,
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title="Number of deals",
    )
    return fig


def plot_monthly_conversion_trends(
    monthly_cohort_df: pl.DataFrame,
    conversion_cols: list[str],
    title: str = "Monthly funnel conversion trends",
    segment_col: str | None = None,
) -> go.Figure:
    index_cols = ["created_month"] + ([segment_col] if segment_col else [])

    plot_df = monthly_cohort_df.select(index_cols + conversion_cols).unpivot(
        on=conversion_cols,
        index=index_cols,
        variable_name="metric",
        value_name="conversion_rate",
    )

    if segment_col:
        fig = px.line(
            plot_df,
            x="created_month",
            y="conversion_rate",
            color=segment_col,
            line_dash="metric",
            markers=True,
            title=title,
        )
    else:
        fig = px.line(
            plot_df,
            x="created_month",
            y="conversion_rate",
            color="metric",
            markers=True,
            title=title,
        )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Creation month",
        yaxis_title="Conversion rate",
    )
    fig.update_yaxes(tickformat=".0%")
    return fig


def plot_monthly_duration_trends(
    monthly_duration_df: pl.DataFrame,
    duration_cols: list[str],
    title: str = "Monthly duration trends",
    segment_col: str | None = None,
) -> go.Figure:
    index_cols = ["created_month"] + ([segment_col] if segment_col else [])

    plot_df = monthly_duration_df.select(index_cols + duration_cols).unpivot(
        on=duration_cols,
        index=index_cols,
        variable_name="metric",
        value_name="days",
    )

    if segment_col:
        fig = px.line(
            plot_df,
            x="created_month",
            y="days",
            color=segment_col,
            line_dash="metric",
            markers=True,
            title=title,
        )
    else:
        fig = px.line(
            plot_df,
            x="created_month",
            y="days",
            color="metric",
            markers=True,
            title=title,
        )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Creation month",
        yaxis_title="Days",
    )
    return fig


def plot_monthly_sql_to_demo_score_rate(
    df: pl.DataFrame,
    title: str = "Monthly SQL → demo score rate",
    segment_col: str | None = None,
) -> go.Figure:
    fig = px.line(
        df,
        x="created_month",
        y="sql_to_demo_score_rate",
        color=segment_col,
        markers=True,
        title=title,
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Creation month",
        yaxis_title="SQL → demo score rate",
    )
    fig.update_yaxes(tickformat=".0%")
    return fig


STAGE_ORDER = ["created", "mql", "sql", "opportunity", "closed_won", "closed_lost"]


def plot_avg_deal_amount_by_stage(
    avg_amounts: dict[str, float | None],
    title: str = "Average deal amount by funnel stage",
) -> go.Figure:
    stages = [s for s in STAGE_ORDER]
    values = [avg_amounts[s] for s in stages]  # type: ignore[index]
    plot_df = pl.DataFrame({"stage": stages, "avg_deal_amount": values})
    fig = px.bar(
        plot_df,
        x="stage",
        y="avg_deal_amount",
        text="avg_deal_amount",
        category_orders={"stage": stages},
        title=title,
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_layout(
        template="plotly_white", xaxis_title="Stage", yaxis_title="Avg deal amount (€)"
    )
    return fig
