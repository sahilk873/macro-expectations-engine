# Market-Implied Macro Engine

A systematic macro research platform that extracts market-implied expectations from prediction markets, compares them to official realized macro outcomes, computes macro surprises, and tests cross-asset reactions and regime-aware allocation strategies.

## Overview

This project bridges prediction markets (Kalshi) with official macro data releases to answer:

1. **What do prediction markets imply about upcoming macro data?**
2. **How do market-implied expectations compare to realized outcomes?**
3. **Do macro surprises drive cross-asset returns?**
4. **Can a regime-aware strategy exploit macro signals?**

## Architecture

```
market-implied-macro-engine/
├── src/macro_engine/          # Core Python library
│   ├── config/                # Settings and configuration
│   ├── events/                # Macro event calendar (CPI, FOMC, NFP, GDP, PCE, etc.)
│   ├── kalshi/                # Kalshi prediction market data fetcher
│   ├── mapping/               # Market-to-event mapping with confidence scoring
│   ├── macro/                 # Official macro data sources (BLS, BEA, FRED)
│   ├── prices/                # ETF price data (yfinance with replaceable provider)
│   ├── expectations/          # Implied probability and distribution extraction
│   ├── surprises/             # Surprise computation and labeling
│   ├── studies/               # Event-study analysis
│   ├── regime/                # Macro regime classification
│   ├── backtest/              # Regime-aware ETF allocation backtest
│   ├── robustness/            # Placebo tests and robustness checks
│   └── report/                # Report generation (tables and figures)
├── scripts/                   # Pipeline execution scripts
├── tests/                     # pytest test suite
├── data/                      # Data storage (kalshi, macro, prices, manual, output)
├── reports/                   # Generated reports (tables, figures, notebooks)
├── pyproject.toml             # Project configuration
├── Makefile                   # Automation targets
└── .env.example               # API key template
```

## Design Principles

- **Separation of concerns**: Market-implied expectations, rate-market proxies, manual expectations, and realized outcomes are kept strictly separate.
- **No lookahead bias**: All pre-event snapshots use only data available before the snapshot time. Regime classifications use only historical data.
- **Confidence scoring**: Every market-event mapping includes a confidence score. Low-confidence mappings are exported for manual review, not silently accepted.
- **Replaceable components**: Price data provider is abstract; yfinance is the default MVP implementation.
- **Limitations disclosed**: Daily ETF data is used for intraday events (FOMC); this limitation is clearly noted.

## Research Methodology (v0.2.0)

### Multi-Factor Surprise Attribution

The `macro_engine.factors` module isolates the marginal "alpha" of macro surprises by controlling for standard risk factors:

$$
R_{i,t} = \alpha_i + \beta_{MKT} R_{MKT,t} + \beta_{RATES} R_{RATES,t} + \beta_{CREDIT} R_{CREDIT,t} + \beta_{DOLLAR} R_{DOLLAR,t} + \gamma_i \text{Surprise}_t + \epsilon_{i,t}
$$

- **Panel regression** with asset fixed effects across event windows
- **Benjamini-Hochberg** false discovery rate control for multiple hypothesis testing
- **Cumulative abnormal returns (CAR)** with bootstrap confidence intervals

### Bayesian Shrinkage & Neyman Confidence Intervals

The `macro_engine.studies.bayesian` module applies empirical Bayes methods to improve small-sample inference:

- **Empirical Bayes shrinkage**: pulls event-type mean returns toward the grand mean, weighting by precision ($1/\sigma^2$)
- **Neyman bootstrap-t confidence intervals**: robust to asymmetric, fat-tailed return distributions
- **Bayes factors**: quantify evidence for / against non-zero event returns
- **Sharpe ratio equivalents**: normalize strategy performance for cross-methodology comparison

### Prediction Market Microstructure

The `macro_engine.microstructure` module evaluates Kalshi market quality metrics:

- **VWAP implied probability**: volume-weighted price discovery
- **Bid-ask spread analysis**: mean, median, volatility of spreads
- **Arbitrage detection**: flags markets where $\text{yes\_bid} + \text{no\_bid} > 1$
- **Market depth ratio**: open interest relative to volume
- **Price discovery ratio**: how quickly the mid-price incorporates new information

### HMM-Based Regime Detection

The `macro_engine.regime.hmm_model` module discovers macro-financial regimes from data rather than relying on heuristic thresholds:

- **Gaussian HMM** fitted on GDP, CPI, unemployment, NFP change, Fed funds rate, term spread, and SPY volatility
- **Automatic state ordering** by economic interpretability (expansion vs contraction)
- **Transition probability matrix** for regime persistence analysis
- **Configurable number of regimes** with comparison to the rule-based classification

### Walk-Forward Backtest Validation

The `macro_engine.backtest.walk_forward` module provides rigorous out-of-sample strategy evaluation:

- **Expanding-window walk-forward splits** mimicking live trading conditions
- **Pooled out-of-sample Sharpe ratio**: the gold standard for strategy evaluation
- **Parameter sensitivity analysis** ($\partial \text{Sharpe} / \partial \text{Param}$) for stability assessment
- **Multi-split cross-validation** to reduce overfitting risk

### Surprise Decomposition

The `macro_engine.surprises.decomposition` module enriches binary surprise signals with information-theoretic measures:

- **Log-odds (logit) decomposition**: maps probability changes to surprise levels
- **Entropy-based uncertainty revision**: $\Delta H = -[p\log p + (1-p)\log(1-p)]$
- **Shannon entropy confidence scoring**: flags low-information predictions

### Confounding Event Control

The `macro_engine.studies.confounding` module addresses the fundamental event study problem of overlapping macro releases:

- **Confounding event detection**: finds co-occurring events within a configurable time window
- **Welch t-test** comparing confounded vs unconfounded return distributions
- **Residualization**: $R = \alpha + \beta \cdot N_{\text{confounds}} + \epsilon$ to isolate the marginal effect

### LaTeX Research Report Generator

The `macro_engine.report.latex_report` module produces a publication-quality academic paper:

- **Full structure**: Abstract, Introduction, Data, Methodology, Empirical Results, Robustness, Conclusion
- **Formatted tables** with significance stars and LaTeX rendering
- **Mathematical equations** for all methodology sections
- **Limitations disclosure** and suggestions for future work

## Data Sources

| Source | Data | Access |
|--------|------|--------|
| **Kalshi** | Prediction market series, events, markets, prices | API key required |
| **BLS** | CPI, nonfarm payrolls, unemployment rate | API key required |
| **BEA** | GDP, PCE price index | Via FRED API |
| **FRED** | Macro/financial time series (200+ series) | API key required |
| **Federal Reserve** | FOMC meeting dates and decisions | Public schedule |
| **yfinance** | ETF prices (SPY, QQQ, IWM, TLT, IEF, HYG, etc.) | Public |

## Quick Start

```bash
# Clone and install
git clone <repo>
cd market-implied-macro-engine
make setup

# Install with HMM support (optional, for regime detection)
pip install -e ".[hmm]"

# Set API keys
cp .env.example .env
# Edit .env with your API keys

# Run full pipeline
make run-all
```

## Research Modules (v0.2.0+)

| Module | Location | Function |
|--------|----------|----------|
| Factor Model | `macro_engine.factors` | Multi-factor surprise attribution with BH correction |
| Bayesian Studies | `macro_engine.studies.bayesian` | Shrinkage estimators, Neyman CIs, Bayes factors |
| Microstructure | `macro_engine.microstructure` | Prediction market quality metrics |
| HMM Regime | `macro_engine.regime.hmm_model` | Data-driven Gaussian HMM regime detection |
| Walk-Forward | `macro_engine.backtest.walk_forward` | OOS backtest validation & parameter sensitivity |
| Surprise Decomposition | `macro_engine.surprises.decomposition` | Log-odds & entropy-based surprise analysis |
| Confounding Control | `macro_engine.studies.confounding` | Confounding event detection & residualization |
| LaTeX Report | `macro_engine.report.latex_report` | Publication-quality academic paper generation |

## Pipeline

| Phase | Script | Output |
|-------|--------|--------|
| 1 | Repo setup | Directory structure, config |
| 2 | `build_event_calendar.py` | Macro event calendar |
| 3 | `fetch_kalshi.py` | Kalshi markets and prices |
| 4 | `build_market_mapping.py` | Market-event mappings |
| 5 | `fetch_macro_data.py` | Official macro time series |
| 6 | `fetch_price_data.py` | ETF price data |
| 7 | `compute_implied_expectations.py` | Pre-event expectation snapshots |
| 8 | `compute_surprises.py` | Raw and standardized surprises |
| 9 | `run_event_studies.py` | Cross-asset event studies |
| 10 | `build_regime_model.py` | Regime classifications |
| 11 | `run_backtest.py` | Regime-aware backtest |
| 12 | `run_robustness_checks.py` | Placebo tests and robustness |
| 13 | `generate_report.py` | Summary tables and figures |
| 14 | **Research modules** | Factor model, Bayesian, microstructure, HMM, walk-forward, confounding, LaTeX report |

## Event Types Covered

- **CPI** (Consumer Price Index) — monthly BLS releases
- **FOMC** (Federal Reserve decisions) — 8 meetings/year
- **NFP** (Nonfarm Payrolls) — monthly BLS releases
- **UNEMPLOYMENT** (Unemployment Rate) — monthly BLS releases
- **GDP** (Gross Domestic Product) — quarterly BEA estimates
- **PCE** (Personal Consumption Expenditures) — monthly BEA releases
- **RECESSION** (Recession probability) — tracked via NBER assessments

## Surprise Labeling

| Event | Above Expectations | Below Expectations |
|-------|-------------------|-------------------|
| CPI/PCE | Inflation hot (risk-off) | Inflation cool (risk-on) |
| NFP | Labor strong (risk-on) | Labor weak (risk-off) |
| Unemployment | Labor weak (risk-off) | Labor strong (risk-on) |
| GDP | Growth strong (risk-on) | Growth weak (risk-off) |
| FOMC | Policy hawkish (risk-off) | Policy dovish (risk-on) |

## Event Study Methodology

1. Identify macro events with prediction-market coverage.
2. Create pre-event expectation snapshots at T-1 day and T-1 hour.
3. Compute implied probabilities from binary market mid-prices.
4. Fetch realized macro outcomes from official sources.
5. Compute raw and standardized surprises.
6. Measure asset returns across windows (1D, 3D, 5D, 10D, 21D).
7. Aggregate by event type, surprise direction, and macro regime.

## Limitations

- **Daily ETF data**: Intraday events (FOMC at 2:00 PM, CPI at 8:30 AM) are measured with daily close-to-close returns, which introduces noise.
- **Kalshi coverage**: Not every historical macro event has a corresponding Kalshi market. The mapping is sparse for earlier periods.
- **API rate limits**: BLS and other free-tier APIs have rate limits; batch processing may be slow.
- **Surrogate data**: When official macro API keys are unavailable, FRED serves as a proxy for BEA data.
- **Simulated prices**: Without yfinance, a deterministic dummy provider is used for development.
- **Research modules (v0.2.0)**: The factor model, HMM, Bayesian, microstructure, walk-forward, confounding, and LaTeX report modules are methodological contributions validated on synthetic data. Empirical calibration on live Kalshi data is pending API key availability.
- **HMM module**: Requires `pip install market-implied-macro-engine[hmm]` for the `hmmlearn` dependency.

## Tests

```bash
make test
# Or: python3 -m pytest
```

165 tests covering:
- Core: probability conversion, bucket parsing, event-time alignment, surprise calculation, return windows, no-lookahead bias, event calendar integrity, market mapping confidence
- **Factor model** (11 tests): panel regression, BH correction, CAR with bootstrap CIs
- **Bayesian shrinkage** (16 tests): empirical Bayes, Neyman CIs, Bayes factors, Sharpe equivalents
- **Microstructure** (16 tests): VWAP, spreads, arbitrage detection, depth ratio, price discovery
- **Confounding control** (8 tests): event detection, Welch t-test, residualization
- **Surprise decomposition** (10 tests): log-odds decomposition, entropy confidence scoring
- **Walk-forward backtest** (8 tests): split generation, OOS Sharpe, parameter sensitivity
- **HMM regime** (16 tests): state ordering, transition matrices, feature scaling, reproducibility

## Key Results

See `reports/tables/` for:
- Summary statistics (event coverage, surprise distributions)
- Event study results (by type, direction, qualitative label)
- Backtest performance (returns, Sharpe, drawdown)
- Robustness checks (placebo p-values)
- **Factor model attribution** (surprise alpha net of market/rates/credit/dollar factors)
- **Microstructure quality** (spreads, arbitrage, depth by event type)
- **HMM regime transitions** (transition probabilities, state comparisons)
- **Walk-forward OOS Sharpe** (pooled OOS performance across splits)
- **Confounding robustness** (confounded vs unconfounded event returns)

## Project Status

Implemented phases: 1-13 ✓
Research modules: 8/8 complete (v0.2.0)
- [x] Multi-factor surprise attribution
- [x] Bayesian shrinkage & Neyman CIs
- [x] Prediction market microstructure
- [x] HMM-based regime detection
- [x] Walk-forward backtest validation
- [x] Surprise decomposition (log-odds & entropy)
- [x] Confounding event control
- [x] LaTeX research report generator

## License

MIT
