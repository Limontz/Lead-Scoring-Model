from dataclasses import dataclass, field
from typing import Any

import polars as pl
import plotly.graph_objects as go

from lead_scoring.analysis.plots import (
    plot_avg_deal_amount_by_stage,
    plot_categorical_conversion,
    plot_closed_lost_reasons,
    plot_conversion_rates,
    plot_duration_distribution,
    plot_numeric_boxplot,
    plot_numeric_histogram,
    plot_segment_conversion_rates,
    plot_segment_stage_volumes,
    plot_stage_volumes,
)
from lead_scoring.analysis.funnel_static import (
    FunnelMetrics,
    FunnelProcessMetrics,
    compute_segment_conversion_long,
    compute_segment_stage_volumes_long,
    summarize_categorical_conversion,
    summarize_categorical_conversion_by_segment,
    summarize_closed_lost_reasons,
    summarize_closed_lost_reasons_by_segment,
    summarize_numeric_by_target,
    summarize_numeric_by_target_and_segment,
)
from lead_scoring.analysis.plots import (
    plot_monthly_conversion_trends,
    plot_monthly_duration_trends,
    plot_monthly_sql_to_demo_score_rate,
    plot_monthly_stage_entries,
)
from lead_scoring.analysis.funnel_temporal import (
    STAGE_MONTH_COLS,
    TEMPORAL_CONVERSION_COLS_DEFAULT,
    TEMPORAL_DURATION_COLS_DEFAULT,
    TemporalSummary,
    build_temporal_summary,
    compute_monthly_created_cohort_metrics,
    compute_monthly_duration_trends,
    compute_monthly_segment_conversion_trends,
    compute_monthly_segment_duration_trends,
    compute_monthly_segment_stage_entries,
    compute_monthly_sql_to_demo_score_rate,
    compute_monthly_stage_entries,
)


@dataclass
class FunnelAnalysisResult:
    full_df: pl.DataFrame
    scoped_df: pl.DataFrame
    step: str | None
    step_label: str
    target_col: str | None
    segment_col: str | None

    overall_metrics: FunnelMetrics
    process_metrics: FunnelProcessMetrics
    segment_metrics: pl.DataFrame | None
    segment_sql_to_demo_score: pl.DataFrame | None
    duration_summary: pl.DataFrame
    segment_duration_summary: pl.DataFrame | None

    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    duration_features: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "step_label": self.step_label,
            "target_col": self.target_col,
            "segment_col": self.segment_col,
            "rows_in_scope": self.scoped_df.height,
            "overall_volumes": self.overall_metrics.volumes,
            "overall_step_conversion_rates": self.overall_metrics.step_conversion_rates,
            "overall_vs_created_conversion_rates": self.overall_metrics.vs_created_conversion_rates,
            "sql_to_demo_score_rate": self.process_metrics.sql_to_demo_score_rate,
            "avg_deal_amount_by_stage": self.overall_metrics.avg_deal_amount_by_stage,
        }

    def available_features(self) -> dict[str, list[str]]:
        return {
            "numeric_features": [
                f for f in self.numeric_features if f in self.scoped_df.columns
            ],
            "categorical_features": [
                f for f in self.categorical_features if f in self.scoped_df.columns
            ],
            "duration_features": [
                f for f in self.duration_features if f in self.full_df.columns
            ],
        }

    def segment_summary(self) -> pl.DataFrame:
        if self.segment_metrics is None or self.segment_col is None:
            raise ValueError("No segment_col was provided.")
        return self.segment_metrics

    def segment_sql_to_demo_score_summary(self) -> pl.DataFrame:
        if self.segment_sql_to_demo_score is None or self.segment_col is None:
            raise ValueError("No segment_col was provided.")
        return self.segment_sql_to_demo_score

    def numeric_summary(self, feature: str) -> pl.DataFrame:
        if self.target_col is None:
            raise ValueError(
                "numeric_summary is only available for step-scoped analyses."
            )

        if self.segment_col is None:
            return summarize_numeric_by_target(self.scoped_df, feature, self.target_col)

        return summarize_numeric_by_target_and_segment(
            self.scoped_df,
            feature,
            self.target_col,
            self.segment_col,
        )

    def categorical_summary(self, feature: str, min_count: int = 30) -> pl.DataFrame:
        if self.target_col is None:
            raise ValueError(
                "categorical_summary is only available for step-scoped analyses."
            )

        if self.segment_col is None:
            return summarize_categorical_conversion(
                self.scoped_df,
                feature,
                self.target_col,
                min_count=min_count,
            )

        return summarize_categorical_conversion_by_segment(
            self.scoped_df,
            feature,
            self.target_col,
            self.segment_col,
            min_count=min_count,
        )

    def closed_lost_reasons(
        self,
        reason_col: str = "DEAL_CLOSED_LOST_REASON",
        top_n: int = 15,
    ) -> pl.DataFrame:
        if self.segment_col is None:
            return summarize_closed_lost_reasons(
                self.scoped_df,
                reason_col=reason_col,
                top_n=top_n,
            )

        return summarize_closed_lost_reasons_by_segment(
            self.scoped_df,
            self.segment_col,
            reason_col=reason_col,
            top_n=top_n,
        )

    def plot_stage_volumes(self) -> go.Figure:
        if self.segment_col is None:
            return plot_stage_volumes(
                self.overall_metrics.volumes,
                title="Funnel volumes",
            )

        if self.segment_metrics is None:
            raise ValueError("No segment_metrics was provided.")
        plot_df = compute_segment_stage_volumes_long(
            self.segment_metrics, self.segment_col
        )
        return plot_segment_stage_volumes(
            plot_df,
            segment_col=self.segment_col,
            title=f"Funnel volumes by {self.segment_col}",
        )

    def plot_step_conversion_rates(self) -> go.Figure:
        if self.segment_col is None:
            return plot_conversion_rates(
                self.overall_metrics.step_conversion_rates,
                title="Step conversion rates",
            )

        if self.segment_metrics is None:
            raise ValueError("No segment_metrics was provided.")
        plot_df = compute_segment_conversion_long(
            self.segment_metrics,
            self.segment_col,
            ["creation_to_mql", "mql_to_sql", "sql_to_opp", "opp_to_won"],
        )
        return plot_segment_conversion_rates(
            plot_df,
            segment_col=self.segment_col,
            title=f"Step conversion rates by {self.segment_col}",
        )

    def plot_vs_created_conversion_rates(self) -> go.Figure:
        if self.segment_col is None:
            return plot_conversion_rates(
                self.overall_metrics.vs_created_conversion_rates,
                title="Created-based conversion rates",
            )

        if self.segment_metrics is None:
            raise ValueError("No segment_metrics was provided.")
        plot_df = compute_segment_conversion_long(
            self.segment_metrics,
            self.segment_col,
            [
                "creation_to_mql",
                "creation_to_sql",
                "creation_to_opportunity",
                "creation_to_won",
                "creation_to_lost",
            ],
        )
        return plot_segment_conversion_rates(
            plot_df,
            segment_col=self.segment_col,
            title=f"Created-based conversion rates by {self.segment_col}",
        )

    def plot_numeric_boxplot(self, feature: str) -> go.Figure:
        if self.target_col is None:
            raise ValueError(
                "plot_numeric_boxplot is only available for step-scoped analyses."
            )

        title = f"{self.step_label} | {feature}"
        if self.segment_col:
            title += f" by {self.segment_col}"

        return plot_numeric_boxplot(
            self.scoped_df,
            feature=feature,
            target_col=self.target_col,
            title=title,
            segment_col=self.segment_col,
        )

    def plot_numeric_histogram(self, feature: str) -> go.Figure:
        if self.target_col is None:
            raise ValueError(
                "plot_numeric_histogram is only available for step-scoped analyses."
            )

        title = f"{self.step_label} | {feature}"
        if self.segment_col:
            title += f" by {self.segment_col}"

        return plot_numeric_histogram(
            self.scoped_df,
            feature=feature,
            target_col=self.target_col,
            title=title,
            segment_col=self.segment_col,
        )

    def plot_duration_distribution(self, feature: str) -> go.Figure:
        if feature not in self.duration_features:
            raise ValueError(
                f"{feature} is not registered as a duration feature. "
                f"Available duration features: {self.duration_features}"
            )
        if feature not in self.full_df.columns:
            raise ValueError(f"{feature} not found in dataframe columns.")

        title = f"Duration distribution | {feature}"
        if self.segment_col:
            title += f" by {self.segment_col}"

        return plot_duration_distribution(
            self.full_df,
            feature=feature,
            title=title,
            segment_col=self.segment_col,
        )

    def plot_categorical_conversion(
        self, feature: str, min_count: int = 30
    ) -> go.Figure:
        if self.target_col is None:
            raise ValueError(
                "plot_categorical_conversion is only available for step-scoped analyses."
            )

        summary_df = self.categorical_summary(feature, min_count=min_count)

        title = f"{self.step_label} | Conversion rate by {feature}"
        if self.segment_col:
            title += f" and {self.segment_col}"

        return plot_categorical_conversion(
            summary_df,
            feature=feature,
            title=title,
            segment_col=self.segment_col,
        )

    def plot_avg_deal_amount_by_stage(self) -> go.Figure:
        return plot_avg_deal_amount_by_stage(
            avg_amounts=self.overall_metrics.avg_deal_amount_by_stage,
            title="Average deal amount by funnel stage",
        )

    def plot_closed_lost_reasons(
        self,
        reason_col: str = "DEAL_CLOSED_LOST_REASON",
        top_n: int = 15,
    ) -> go.Figure:
        reason_df = self.closed_lost_reasons(reason_col=reason_col, top_n=top_n)
        if reason_df.is_empty():
            raise ValueError("No closed lost reasons available in the current scope.")

        title = f"{self.step_label} | Closed lost reasons"
        if self.segment_col:
            title += f" by {self.segment_col}"

        return plot_closed_lost_reasons(
            reason_df,
            reason_col=reason_col,
            title=title,
            segment_col=self.segment_col,
        )


@dataclass
class TemporalAnalysisResult:
    full_df: pl.DataFrame
    temporal_df: pl.DataFrame
    segment_col: str | None

    summary_stats: TemporalSummary
    monthly_stage_entries: pl.DataFrame
    monthly_created_cohort: pl.DataFrame
    monthly_duration_trends: pl.DataFrame
    monthly_sql_to_demo_score: pl.DataFrame

    monthly_segment_stage_entries: pl.DataFrame | None
    monthly_segment_created_cohort: pl.DataFrame | None
    monthly_segment_duration_trends: pl.DataFrame | None

    conversion_cols: list[str] = field(default_factory=list)
    duration_cols: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "segment_col": self.segment_col,
            "n_rows": self.temporal_df.height,
            "n_months": self.summary_stats.n_months,
            "first_month": self.summary_stats.first_month,
            "last_month": self.summary_stats.last_month,
        }

    def available_features(self) -> dict[str, list[str]]:
        return {
            "conversion_features": [c for c in self.conversion_cols if c in self.monthly_created_cohort.columns],
            "duration_features": [c for c in self.duration_cols if c in self.monthly_duration_trends.columns],
        }

    def segment_summary(self) -> pl.DataFrame:
        if self.monthly_segment_created_cohort is None or self.segment_col is None:
            raise ValueError("No segment_col was provided.")
        return self.monthly_segment_created_cohort

    def plot_stage_entries(self) -> go.Figure:
        if self.segment_col is None:
            return plot_monthly_stage_entries(
                self.monthly_stage_entries,
                title="Monthly funnel stage entries",
            )

        if self.monthly_segment_stage_entries is None:
            raise ValueError("No monthly stage entries available.")
        return plot_monthly_stage_entries(
            self.monthly_segment_stage_entries,
            title=f"Monthly funnel stage entries by {self.segment_col}",
            segment_col=self.segment_col,
        )

    def plot_conversion_trends(self, conversion_cols: list[str] | None = None) -> go.Figure:
        conversion_cols = conversion_cols or self.conversion_cols

        if self.segment_col is None:
            return plot_monthly_conversion_trends(
                self.monthly_created_cohort,
                conversion_cols=conversion_cols,
                title="Monthly funnel conversion trends",
            )

        if self.monthly_segment_created_cohort is None:
            raise ValueError("No monthly created cohort data available.")
        return plot_monthly_conversion_trends(
            self.monthly_segment_created_cohort,
            conversion_cols=conversion_cols,
            title=f"Monthly funnel conversion trends by {self.segment_col}",
            segment_col=self.segment_col,
        )

    def plot_duration_trends(self, duration_cols: list[str] | None = None) -> go.Figure:
        duration_cols = duration_cols or self.duration_cols

        if self.segment_col is None:
            return plot_monthly_duration_trends(
                self.monthly_duration_trends,
                duration_cols=duration_cols,
                title="Monthly duration trends",
            )

        if self.monthly_segment_duration_trends is None:
            raise ValueError("No monthly duration trends available.")
        return plot_monthly_duration_trends(
            self.monthly_segment_duration_trends,
            duration_cols=duration_cols,
            title=f"Monthly duration trends by {self.segment_col}",
            segment_col=self.segment_col,
        )

    def plot_sql_to_demo_score_trends(self) -> go.Figure:
        return plot_monthly_sql_to_demo_score_rate(
            self.monthly_sql_to_demo_score,
            title="Monthly SQL → demo score rate",
            segment_col=self.segment_col,
        )