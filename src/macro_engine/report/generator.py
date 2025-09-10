from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from macro_engine.config.settings import EngineConfig, get_settings


def generate_summary_table(
    surprises: pd.DataFrame,
    event_studies: pd.DataFrame,
    config: Optional[EngineConfig] = None,
) -> str:
    """Generate a summary table of event studies."""
    if surprises.empty:
        return "No surprise data available."

    lines = []
    lines.append("=" * 100)
    lines.append("MARKET-IMPLIED MACRO ENGINE: SUMMARY STATISTICS")
    lines.append("=" * 100)
    lines.append("")

    # 1. Event coverage
    lines.append("--- Event Coverage ---")
    coverage = (
        surprises.groupby("event_type")
        .agg(
            n_events=("event_id", "nunique"),
            n_markets=("market_ticker", "nunique"),
        )
        .reset_index()
    )
    for _, row in coverage.iterrows():
        lines.append(
            f"  {row['event_type']:15s}  {int(row['n_events']):4d} events  "
            f"{int(row['n_markets']):4d} markets"
        )
    lines.append("")

    # 2. Surprise statistics
    lines.append("--- Surprise Statistics ---")
    for event_type in surprises["event_type"].unique():
        subset = surprises[surprises["event_type"] == event_type]
        raw = subset["raw_surprise"].dropna()
        std = subset["standardized_surprise"].dropna()

        if len(raw) > 0:
            lines.append(
                f"  {event_type:15s}  Raw: mean={raw.mean():+.4f}  std={raw.std():.4f}  "
                f"|  Std: mean={std.mean():+.3f}  std={std.std():.3f}  n={len(raw)}"
            )
    lines.append("")

    # 3. Direction breakdown
    lines.append("--- Surprise Direction ---")
    direction_counts = surprises.groupby(["event_type", "direction"]).size().unstack(fill_value=0)
    for idx in direction_counts.index:
        row = direction_counts.loc[idx]
        parts = [f"{col}: {int(row[col])}" for col in direction_counts.columns]
        lines.append(f"  {idx:15s}  {', '.join(parts)}")
    lines.append("")

    # 4. Event study summary
    if not event_studies.empty:
        lines.append("--- Event Study: Average Returns ---")
        return_cols = [c for c in event_studies.columns if c.startswith("return_")]
        if return_cols:
            summary = event_studies.groupby("event_type")[return_cols].mean().round(4)
            for idx in summary.index:
                parts = [f"{col}: {summary.loc[idx, col]:+.4f}" for col in return_cols]
                lines.append(f"  {idx:15s}  {', '.join(parts)}")

    return "\n".join(lines)


def generate_event_study_table(
    event_studies: pd.DataFrame,
    config: Optional[EngineConfig] = None,
) -> str:
    """Generate detailed event study results table."""
    if event_studies.empty:
        return "No event study data available."

    from macro_engine.studies.event_study import aggregate_event_study

    # Aggregate by event type
    by_type = aggregate_event_study(event_studies, group_by=["event_type"])
    # Aggregate by event type and direction
    by_direction = aggregate_event_study(event_studies, group_by=["event_type", "direction"])
    # Aggregate by event type and qualitative label
    by_qual = aggregate_event_study(event_studies, group_by=["event_type", "qualitative"])

    lines = []
    lines.append("=" * 120)
    lines.append("EVENT STUDY ANALYSIS")
    lines.append("=" * 120)
    lines.append("")

    for title, df in [
        ("By Event Type", by_type),
        ("By Event Type & Direction", by_direction),
        ("By Event Type & Qualitative", by_qual),
    ]:
        lines.append(f"--- {title} ---")
        if not df.empty:
            lines.append(df.to_string(index=False))
        else:
            lines.append("  (no data)")
        lines.append("")

    return "\n".join(lines)


def generate_backtest_table(
    backtest_results: pd.DataFrame,
    config: Optional[EngineConfig] = None,
) -> str:
    """Generate backtest performance table."""
    from macro_engine.backtest.strategy import compute_performance_metrics

    metrics = compute_performance_metrics(backtest_results)
    if not metrics:
        return "No backtest data available."

    lines = []
    lines.append("=" * 60)
    lines.append("BACKTEST PERFORMANCE")
    lines.append("=" * 60)

    # SPY benchmark
    spy_return = None
    if "weight_SPY" in backtest_results.columns:
        first_val = backtest_results["portfolio_value"].iloc[0]
        last_val = backtest_results["portfolio_value"].iloc[-1]
        total_ret = last_val / first_val - 1.0
        spy_return = total_ret

    lines.append(f"  Total Return:           {metrics['total_return']:>8.2%}")
    lines.append(f"  Annualized Return:      {metrics['annualized_return']:>8.2%}")
    lines.append(f"  Annualized Volatility:  {metrics['annualized_volatility']:>8.2%}")
    lines.append(f"  Sharpe Ratio:           {metrics['sharpe_ratio']:>8.3f}")
    lines.append(f"  Maximum Drawdown:       {metrics['max_drawdown']:>8.2%}")
    lines.append(f"  Calmar Ratio:           {metrics['calmar_ratio']:>8.3f}")
    lines.append(f"  Win Rate:               {metrics['win_rate']:>8.2%}")
    lines.append(f"  Avg Turnover:           {metrics['avg_turnover']:>8.2%}")
    if spy_return is not None:
        lines.append(f"  SPY Buy & Hold:         {spy_return:>8.2%}")
    lines.append(f"  N Periods:              {metrics['n_periods']:>8d}")

    return "\n".join(lines)


def generate_robustness_table(
    robustness_results: dict[str, pd.DataFrame],
    config: Optional[EngineConfig] = None,
) -> str:
    """Generate robustness checks table."""
    lines = []
    lines.append("=" * 80)
    lines.append("ROBUSTNESS CHECKS")
    lines.append("=" * 80)
    lines.append("")

    # Placebo summary
    placebo_summary = robustness_results.get("placebo_summary", pd.DataFrame())
    if not placebo_summary.empty:
        lines.append("--- Placebo Test Results ---")
        lines.append("  Compares actual event-study returns vs. placebo distribution.")
        lines.append("")
        for _, row in placebo_summary.iterrows():
            sig = "***" if row.get("significant_5pct", False) else ""
            lines.append(
                f"  {row['window']:15s}  "
                f"Actual: {row['actual_mean_return']:+.4f}  "
                f"p-value: {row['p_value']:.4f}  "
                f"n_placebo: {int(row['placebo_n_obs']):5d}  "
                f"{sig}"
            )

    # Placebo date and sign iteration counts
    for key in ["placebo_dates", "placebo_signs"]:
        df = robustness_results.get(key, pd.DataFrame())
        if not df.empty:
            n_iter = df["iteration"].nunique()
            n_obs = len(df)
            lines.append(f"  {key:20s}: {n_iter} iterations, {n_obs} total observations")

    lines.append("")
    lines.append("  Significance: *** p < 0.05")
    return "\n".join(lines)


def _save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def generate_all_tables(
    surprises: pd.DataFrame,
    event_studies: pd.DataFrame,
    backtest_results: pd.DataFrame,
    robustness_results: dict[str, pd.DataFrame],
    config: Optional[EngineConfig] = None,
) -> dict[str, Path]:
    """Generate and save all summary tables."""
    cfg = config or get_settings()
    paths = {}

    summary = generate_summary_table(surprises, event_studies, config)
    p = cfg.tables_dir / "summary_table.txt"
    _save_text(p, summary)
    paths["summary"] = p

    es = generate_event_study_table(event_studies, config)
    p = cfg.tables_dir / "event_study_table.txt"
    _save_text(p, es)
    paths["event_study"] = p

    bt = generate_backtest_table(backtest_results, config)
    p = cfg.tables_dir / "backtest_table.txt"
    _save_text(p, bt)
    paths["backtest"] = p

    rb = generate_robustness_table(robustness_results, config)
    p = cfg.tables_dir / "robustness_table.txt"
    _save_text(p, rb)
    paths["robustness"] = p

    return paths


def generate_all_figures(
    surprises: pd.DataFrame,
    event_studies: pd.DataFrame,
    backtest_results: pd.DataFrame,
    config: Optional[EngineConfig] = None,
) -> dict[str, str]:
    """Generate all figures (placeholder for notebook-based generation)."""
    cfg = config or get_settings()
    figures = {}

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1. Surprise distribution
        if not surprises.empty and "standardized_surprise" in surprises.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            agg = surprises.groupby("event_type")["standardized_surprise"].agg(
                ["mean", "std", "count"]
            )
            ax.bar(agg.index, agg["mean"], yerr=agg["std"], capsize=5)
            ax.set_title("Standardized Surprises by Event Type (mean ± std)")
            ax.set_ylabel("Standardized Surprise")
            path = str(cfg.figures_dir / "surprise_distribution.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            figures["surprise_distribution"] = path

        # 2. Event study returns heatmap
        if not event_studies.empty:
            return_cols = [c for c in event_studies.columns if c.startswith("return_")]
            if return_cols:
                heatmap_data = (
                    event_studies.groupby(["event_type", "ticker"])[return_cols[0]].mean().unstack()
                )
                if not heatmap_data.empty:
                    fig, ax = plt.subplots(figsize=(14, 8))
                    im = ax.imshow(
                        heatmap_data.values, cmap="RdYlGn", aspect="auto", vmin=-0.03, vmax=0.03
                    )
                    ax.set_xticks(range(len(heatmap_data.columns)))
                    ax.set_xticklabels(heatmap_data.columns, rotation=45, ha="right")
                    ax.set_yticks(range(len(heatmap_data.index)))
                    ax.set_yticklabels(heatmap_data.index)
                    ax.set_title(f"Average {return_cols[0]} Return by Event Type and Ticker")
                    plt.colorbar(im, ax=ax, label="Return")
                    path = str(cfg.figures_dir / "event_study_heatmap.png")
                    fig.savefig(path, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    figures["event_study_heatmap"] = path

        # 3. Backtest equity curve
        if not backtest_results.empty and "portfolio_value" in backtest_results.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(
                pd.to_datetime(backtest_results["date"]),
                backtest_results["portfolio_value"],
                label="Regime-Aware Strategy",
            )
            ax.set_title("Backtest: Portfolio Value Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Portfolio Value (log scale)")
            ax.set_yscale("log")
            ax.legend()
            ax.grid(True, alpha=0.3)
            path = str(cfg.figures_dir / "backtest_equity_curve.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            figures["backtest_equity_curve"] = path

    except ImportError:
        pass

    return figures


def generate_research_report(
    config: Optional[EngineConfig] = None,
) -> str:
    """Generate an empty research report placeholder."""
    return (
        "# Research Report: Market-Implied Macro Engine\n\n"
        "This report will be populated with:\n"
        "- Data description\n"
        "- Event study methodology\n"
        "- Empirical results\n"
        "- Backtest performance\n"
        "- Robustness checks\n"
        "- Conclusions and limitations\n"
    )
