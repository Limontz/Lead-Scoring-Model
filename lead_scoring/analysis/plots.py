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
