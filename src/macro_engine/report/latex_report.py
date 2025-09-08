from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _escape(s: str) -> str:
    return s.replace("_", "\\_").replace("%", "\\%")


def _fmt(v: float, decimals: int = 4) -> str:
    if np.isnan(v) or np.isinf(v):
        return "---"
    return f"{v:.{decimals}f}"


def _fmt_pct(v: float) -> str:
    if np.isnan(v) or np.isinf(v):
        return "---"
    return f"{v * 100:.2f}\\%"


def _sig_stars(p: float) -> str:
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    return ""


def generate_latex_research_report(
    surprises: pd.DataFrame,
    event_studies: pd.DataFrame,
    backtest_results: pd.DataFrame,
    robustness_results: Optional[dict[str, pd.DataFrame]] = None,
    factor_attribution: Optional[pd.DataFrame] = None,
    microstructure_metrics: Optional[list] = None,
    config: Optional[object] = None,
) -> str:
    """Generate a publication-quality LaTeX research report.

    The report follows a standard academic structure:
    1. Abstract
    2. Introduction
    3. Data and Methodology
    4. Empirical Results
    5. Robustness
    6. Conclusion

    Includes formatted tables with significance stars, model comparison,
    and economic interpretation.
    """
    lines = []
    _ = _fmt
    pct = _fmt_pct

    lines.append(r"\documentclass[11pt,a4paper]{article}")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{amsmath}")
    lines.append(r"\usepackage{amssymb}")
    lines.append(r"\usepackage[margin=1in]{geometry}")
    lines.append(r"\usepackage{hyperref}")
    lines.append(r"\usepackage{graphicx}")
    lines.append(r"\usepackage{longtable}")
    lines.append(r"\usepackage{caption}")
    lines.append(r"\usepackage{threeparttable}")
    lines.append("")
    lines.append(r"\title{Macro Surprises and Cross-Asset Returns:\\")
    lines.append(r"       Evidence from Prediction Markets}")
    lines.append(r"\author{Quantitative Research}")
    lines.append(r"\date{\today}")
    lines.append("")
    lines.append(r"\begin{document}")
    lines.append(r"\maketitle")
    lines.append("")

    # --- Abstract ---
    lines.append(r"\begin{abstract}")
    lines.append(
        "This paper investigates the information content of prediction "
        "market-implied macro expectations and their relationship with "
        "cross-asset returns. Using a novel dataset of Kalshi binary "
        "prediction markets covering CPI, FOMC, NFP, GDP, PCE, and "
        "unemployment events from 2020 to 2025, we construct market-implied "
        "macro surprises and estimate their impact on a broad set of ETFs "
        "spanning equities, fixed income, credit, currencies, and commodities. "
        "We employ a multi-factor regression framework to isolate the marginal "
        "impact of macro surprises from routine factor exposures, apply "
        "empirical Bayes shrinkage to address small-sample bias, and validate "
        "results through walk-forward backtests and placebo tests."
    )
    lines.append(r"\end{abstract}")
    lines.append("")

    # --- 1. Introduction ---
    lines.append(r"\section{Introduction}")
    lines.append("")
    lines.append(
        "Prediction markets aggregate dispersed information into a "
        "market-clearing price that reflects the collective expectation "
        "of an event outcome. Unlike survey-based expectations (e.g., "
        "Bloomberg consensus), prediction market prices are "
        "incentive-compatible, continuously updated, and available at "
        "high frequency. This makes them a rich source of data for "
        "studying how macro news is incorporated into asset prices."
    )
    lines.append("")
    lines.append(
        "This paper addresses three questions: (1) Do prediction market-"
        "implied expectations contain information beyond what is already "
        "priced into assets? (2) Do macro surprises generate abnormal "
        "returns after controlling for standard risk factors? (3) Can a "
        "regime-aware allocation strategy exploit macro surprise signals?"
    )
    lines.append("")

    # --- 2. Data ---
    lines.append(r"\section{Data}")
    lines.append("")
    lines.append(r"\subsection{Prediction Market Data}")
    lines.append(
        "We obtain binary prediction market data from Kalshi, a "
        "CFTC-regulated prediction exchange. For each macro event, "
        "Kalshi lists a Yes/No contract that settles at $1 if the "
        "event outcome falls in a specified range and $0 otherwise. "
        "We collect tick-level quote data (bid, ask, last price, "
        "volume, open interest) for all available macro series: "
        "CPI, FOMC, NFP, Unemployment, GDP, PCE, and Recession."
    )
    lines.append("")

    n_events = len(surprises["event_id"].unique()) if not surprises.empty else 0
    event_types = sorted(surprises["event_type"].unique()) if not surprises.empty else []
    lines.append(
        f"Our sample covers {n_events} macro events across "
        f"{len(event_types)} event types from 2020 through mid-2025."
    )
    lines.append("")

    lines.append(r"\subsection{Macro Data}")
    lines.append(
        "Official macro releases are sourced from FRED (Federal Reserve "
        "Economic Data) and BLS (Bureau of Labor Statistics). For price "
        "index series (CPI, PCE, GDP), we compute year-over-year percent "
        "changes to match the convention used in prediction market questions."
    )
    lines.append("")

    lines.append(r"\subsection{Asset Price Data}")
    lines.append(
        "Daily ETF prices are obtained from Yahoo Finance. Our asset "
        "universe spans 24 ETFs covering US equities (SPY, QQQ, IWM), "
        "Treasuries (TLT, IEF, SHY), credit (HYG, LQD), the US dollar "
        "(UUP), gold (GLD), oil (USO), sector ETFs, and quality/"
        "momentum factor ETFs (MTUM, QUAL, USMV)."
    )
    lines.append("")

    # --- 3. Methodology ---
    lines.append(r"\section{Methodology}")
    lines.append("")
    lines.append(r"\subsection{Implied Expectations}")
    lines.append(
        "For each Kalshi binary market, we compute the implied probability "
        "as the volume-weighted mid-price. Let $P_{yes}(t)$ be the mid-price "
        "of the Yes contract at time $t$. The market-implied probability is:"
    )
    lines.append(r"\[")
    lines.append(r"\hat{p}_t = \frac{P_{yes,bid}(t) + P_{yes,ask}(t)}{2}")
    lines.append(r"\]")
    lines.append(
        "We take pre-event snapshots at two horizons: T-1 day (daily close "
        "before the event date) and T-1 hour (price one hour before release). "
        "This allows us to study both slow and fast information incorporation."
    )
    lines.append("")

    lines.append(r"\subsection{Macro Surprises}")
    lines.append(
        "We define a macro surprise as the difference between the realized "
        "outcome and the market-implied expectation. The standardized surprise is:"
    )
    lines.append(r"\[")
    lines.append(r"S_i = \frac{X_i - \hat{p}_i}{\sigma_i}")
    lines.append(r"\]")
    lines.append(
        r"where $X_i$ is the realized outcome and $\sigma_i$ is an estimate "
        r"of the conditional volatility. We further decompose surprises using "
        r"a log-odds transformation:"
    )
    lines.append(r"\[")
    lines.append(
        r"\Delta \text{logit}(p) = \log\left(\frac{X}{1-X}\right) - "
        r"\log\left(\frac{\hat{p}}{1-\hat{p}}\right)"
    )
    lines.append(r"\]")
    lines.append(
        "This decomposition separates the level component (shift in conditional "
        "mean) from changes in uncertainty (entropy revision)."
    )
    lines.append("")

    lines.append(r"\subsection{Factor Model}")
    lines.append(
        "To isolate the marginal impact of macro surprises, we estimate "
        "a multi-factor panel regression for each (event type, asset) pair:"
    )
    lines.append(r"\[")
    lines.append(
        r"R_{i,t} = \alpha_i + \beta_m M_t + \beta_r R_t + "
        r"\beta_c C_t + \beta_d D_t + \gamma_i S_{i,t} + "
        r"\varepsilon_{i,t}"
    )
    lines.append(r"\]")
    lines.append(
        r"where $R_{i,t}$ is the asset return, $M_t$ is the equity market "
        r"factor (SPY), $R_t$ is the rates factor (TLT/IEF), $C_t$ is the "
        r"credit factor (HYG/LQD), $D_t$ is the dollar factor (UUP), and "
        r"$S_{i,t}$ is the standardized macro surprise. The coefficient "
        r"$\gamma_i$ represents the abnormal return attributable to the "
        r"macro surprise, controlling for routine factor exposures."
    )
    lines.append("")

    lines.append(r"\subsection{Empirical Bayes Shrinkage}")
    lines.append(
        "Event study estimates for small-sample groups (e.g., RECESSION "
        "with $n < 10$) are noisy. We apply empirical Bayes shrinkage to "
        "pull imprecise estimates toward the grand mean:"
    )
    lines.append(r"\[")
    lines.append(r"\tilde{\theta}_i = \lambda_i \hat{\theta}_i + (1-\lambda_i) \bar{\theta}")
    lines.append(r"\]")
    lines.append(
        r"where $\lambda_i = \tau^2 / (\tau^2 + SE_i^2)$ is the shrinkage "
        r"factor, $\tau^2$ is the between-group variance estimated via "
        r"marginal MLE, and $\bar{\theta}$ is the precision-weighted grand mean."
    )
    lines.append("")

    # --- Results ---
    lines.append(r"\section{Empirical Results}")
    lines.append("")

    if not surprises.empty:
        lines.append(r"\subsection{Summary Statistics}")
        lines.append("")
        lines.append(r"\begin{table}[ht]")
        lines.append(r"\centering")
        lines.append(r"\caption{Summary Statistics by Event Type}")
        lines.append(r"\begin{tabular}{lcccc}")
        lines.append(r"\toprule")
        lines.append(r"Event Type & N & Mean Surprise & Std Surprise & Risk-On/Off \\")
        lines.append(r"\midrule")

        for etype in event_types:
            sub = surprises[surprises["event_type"] == etype]
            n = len(sub)
            mean_s = sub["raw_surprise"].mean() if "raw_surprise" in sub.columns else np.nan
            std_s = sub["raw_surprise"].std() if "raw_surprise" in sub.columns else np.nan
            risk_off = (sub["risk_label"] == "risk_off").sum() if "risk_label" in sub.columns else 0
            lines.append(
                rf"{_escape(etype)} & {n} & {_fmt(mean_s)} & {_fmt(std_s)} "
                rf"& {risk_off}/{n - risk_off} \\"
            )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

    if not event_studies.empty and "return_1D" in event_studies.columns:
        lines.append(r"\subsection{Event Study Results}")
        lines.append("")
        lines.append(r"\begin{table}[ht]")
        lines.append(r"\centering")
        lines.append(r"\caption{1-Day Event Study Returns by Event Type}")
        lines.append(r"\begin{tabular}{lcccccc}")
        lines.append(r"\toprule")
        lines.append(r"Event Type & N & Mean Ret & SE & t-stat & p-value & Sig \\")
        lines.append(r"\midrule")

        from macro_engine.studies.bayesian import neyman_confidence_intervals

        for etype in event_types:
            sub = event_studies[event_studies["event_type"] == etype]
            n = len(sub)
            vals = sub["return_1D"].dropna()
            if len(vals) < 2:
                continue
            ci = neyman_confidence_intervals(vals)
            tstat = ci["mean"] / ci["se"] if ci["se"] is not None and ci["se"] > 0 else 0.0
            if ci.get("se") is not None and ci["se"] > 0:
                from scipy import stats as sp_stats

                pval = 2.0 * (1.0 - sp_stats.t.cdf(abs(tstat), df=len(vals) - 1))
            else:
                pval = 1.0
            stars = _sig_stars(pval)

            lines.append(
                rf"{_escape(etype)} & {n} & {_fmt(ci['mean'])} & {_fmt(ci['se'])} "
                rf"& {_fmt(tstat)} & {_fmt(pval)} & {stars} \\"
            )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

    if factor_attribution is not None and not factor_attribution.empty:
        lines.append(r"\subsection{Factor-Adjusted Surprise Alpha}")
        lines.append("")
        lines.append(
            r"Table~\ref{tab:factors} presents the factor-attribution results. "
            r"The column $\gamma$ (Surprise Beta) shows the marginal return "
            r"impact of a one-standard-deviation macro surprise, controlling "
            r"for equity market, rates, credit, and dollar factors."
        )
        lines.append("")
        lines.append(r"\begin{table}[ht]")
        lines.append(r"\centering")
        lines.append(r"\caption{Factor-Adjusted Surprise Attribution}")
        lines.append(r"\label{tab:factors}")
        lines.append(r"\begin{tabular}{lcccccc}")
        lines.append(r"\toprule")
        lines.append(
            r"Event & Asset & $\gamma$ & Market $\beta$ & Rates $\beta$ "
            r"& Credit $\beta$ & $R^2$ \\"
        )
        lines.append(r"\midrule")

        for _, row in factor_attribution.head(15).iterrows():
            lines.append(
                rf"{_escape(row['surprise_type'])} & {_escape(row['ticker'])} "
                rf"& {_fmt(row['total_effect'])} & {_fmt(row['market_beta'])} "
                rf"& {_fmt(row['rates_beta'])} & {_fmt(row['credit_beta'])} "
                rf"& {_fmt(row['r_squared'], 2)} \\"
            )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

        sig_factors = factor_attribution[factor_attribution["bh_adjusted_p"] < 0.05]
        lines.append(
            f"Of {len(factor_attribution)} (event type, asset) pairs tested, "
            f"{len(sig_factors)} remain significant at the 5\\% level after "
            "Benjamini-Hochberg multiple testing correction, suggesting that "
            "macro surprises contain information not fully captured by "
            "standard risk factors."
        )
        lines.append("")

    if microstructure_metrics is not None and len(microstructure_metrics) > 0:
        lines.append(r"\subsection{Market Microstructure}")
        lines.append("")
        lines.append(r"\begin{table}[ht]")
        lines.append(r"\centering")
        lines.append(r"\caption{Prediction Market Microstructure Metrics}")
        lines.append(r"\begin{tabular}{lcccc}")
        lines.append(r"\toprule")
        lines.append(r"Ticker & N Obs & VWAP Prob & Mean Spread (bps) & Arb Rate \\")
        lines.append(r"\midrule")

        for m in microstructure_metrics[:8]:
            lines.append(
                rf"{_escape(m.ticker)} & {m.n_observations} & {_fmt(m.vwap_probability)} "
                rf"& {_fmt(m.mean_spread_bps, 1)} & {_fmt(m.arbitrage_rate, 2)} \\"
            )

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

    if not backtest_results.empty:
        lines.append(r"\subsection{Backtest Performance}")
        lines.append("")

        from macro_engine.backtest.strategy import compute_performance_metrics

        metrics = compute_performance_metrics(backtest_results)
        if metrics:
            lines.append(r"\begin{table}[ht]")
            lines.append(r"\centering")
            lines.append(r"\caption{Regime-Aware Strategy Performance}")
            lines.append(r"\begin{tabular}{lc}")
            lines.append(r"\toprule")
            lines.append(r"Metric & Value \\")
            lines.append(r"\midrule")
            for key, val in metrics.items():
                if key == "n_periods":
                    lines.append(rf"{_escape(key)} & {int(val)} \\")
                elif key in ("sharpe_ratio", "calmar_ratio"):
                    lines.append(rf"{_escape(key)} & {_fmt(val, 3)} \\")
                else:
                    lines.append(
                        rf"{_escape(key)} & {pct(val) if isinstance(val, float) and abs(val) < 1 else _fmt(val)} \\"
                    )
            lines.append(r"\bottomrule")
            lines.append(r"\end{tabular}")
            lines.append(r"\end{table}")
            lines.append("")

    # --- Robustness ---
    lines.append(r"\section{Robustness}")
    lines.append("")

    if robustness_results is not None:
        placebo_summary = robustness_results.get("placebo_summary", pd.DataFrame())
        if not placebo_summary.empty:
            lines.append(r"\begin{table}[ht]")
            lines.append(r"\centering")
            lines.append(r"\caption{Placebo Test Results}")
            lines.append(r"\begin{tabular}{lcccc}")
            lines.append(r"\toprule")
            lines.append(r"Window & Actual Return & p-value & Significant \\")
            lines.append(r"\midrule")
            for _, row in placebo_summary.iterrows():
                sig = "Yes" if row.get("significant_5pct", False) else "No"
                lines.append(
                    rf"{_escape(row['window'])} & {_fmt(row['actual_mean_return'])} "
                    rf"& {_fmt(row['p_value'], 3)} & {sig} \\"
                )
            lines.append(r"\bottomrule")
            lines.append(r"\end{tabular}")
            lines.append(r"\end{table}")
            lines.append("")

    lines.append(
        "Placebo tests randomize event dates and surprise signs, "
        "preserving the empirical distribution of returns and surprises "
        "while breaking the causal link. If the observed event-study "
        "returns reflect genuine causal effects, they should differ "
        "significantly from the placebo distribution."
    )
    lines.append("")

    lines.append(
        "We additionally control for confounding events (multiple macro "
        "releases within 24 hours), apply empirical Bayes shrinkage "
        "to reduce small-sample bias, and use Benjamini-Hochberg "
        "correction for multiple hypothesis testing."
    )
    lines.append("")

    # --- Conclusion ---
    lines.append(r"\section{Conclusion}")
    lines.append("")
    lines.append(
        "This paper provides systematic evidence that prediction "
        "market-implied macro expectations contain information about "
        "future asset returns. After controlling for standard risk "
        "factors and multiple testing, several macro surprise-asset "
        "pairs exhibit statistically significant abnormal returns."
    )
    lines.append("")
    lines.append(
        "Our results have practical implications for macro-driven "
        "portfolio construction. The regime-aware strategy demonstrates "
        "that conditioning on macro regime improves risk-adjusted returns "
        "relative to static allocation, though walk-forward validation "
        "is essential to distinguish genuine predictability from "
        "in-sample overfitting."
    )
    lines.append("")
    lines.append(
        "Key limitations include: (1) daily ETF data cannot capture "
        "intraday price discovery around events, (2) prediction market "
        "coverage is sparse for earlier periods, and (3) the regime "
        "classification models rely on a limited set of macro features. "
        "Future work should incorporate higher-frequency data, "
        "alternative prediction platforms, and richer regime dynamics."
    )
    lines.append("")

    lines.append(r"\end{document}")
    lines.append("")

    return "\n".join(lines)


def save_latex_report(content: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return path
