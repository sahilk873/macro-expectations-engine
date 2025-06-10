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

# Set API keys
cp .env.example .env
# Edit .env with your API keys

# Run full pipeline
make run-all
```

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

## Tests

```bash
make test
```

Tests cover:
- Probability conversion from binary markets
- Bucket range parsing for multi-bucket markets
- Event-time alignment and pre-event snapshots
- Surprise calculation (raw, standardized, percentage)
- Surprise labeling (inflation, employment, growth, policy)
- Return window computation
- No-lookahead bias verification
- Event calendar integrity
- Market mapping confidence scoring

## Key Results

See `reports/tables/` for:
- Summary statistics (event coverage, surprise distributions)
- Event study results (by type, direction, qualitative label)
- Backtest performance (returns, Sharpe, drawdown)
- Robustness checks (placebo p-values)

## Project Status

Implemented phases: 1-13 ✓

## License

MIT
