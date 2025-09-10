"""Report generation for tables, figures, and LaTeX research reports."""

from macro_engine.report.generator import (
    generate_all_figures,
    generate_all_tables,
    generate_backtest_table,
    generate_event_study_table,
    generate_research_report,
    generate_robustness_table,
    generate_summary_table,
)
from macro_engine.report.latex_report import (
    generate_latex_research_report,
    save_latex_report,
)

__all__ = [
    "generate_summary_table",
    "generate_event_study_table",
    "generate_backtest_table",
    "generate_robustness_table",
    "generate_all_tables",
    "generate_all_figures",
    "generate_research_report",
    "generate_latex_research_report",
    "save_latex_report",
]
