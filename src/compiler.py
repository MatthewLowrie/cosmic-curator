
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from typing import Dict, Any
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html
from src.config import PipelineConfig, logger


def chart_html(figure: go.Figure, include_library: bool = False) -> str:
    """Wraps Plotly objects and parses them cleanly into pure HTML strings."""
    return to_html(
        figure, full_html=False, include_plotlyjs=include_library,
        config={"displaylogo": False, "responsive": True}
    )


def generate_candidates_table(candidates_df: pd.DataFrame) -> str:
    """Processes candidate predictions and compiles the Top-12 HTML Table."""
    top_candidates = candidates_df.sort_values("planet_likeness", ascending=False).head(12)
    formatted_table = top_candidates[[
        "kepoi_name", "planet_likeness", "koi_period", "koi_prad", "koi_teq", "koi_model_snr"
    ]].rename(columns={
        "kepoi_name": "KOI",
        "planet_likeness": "Planet likeness",
        "koi_period": "Period days",
        "koi_prad": "Radius Earths",
        "koi_teq": "Temperature K",
        "koi_model_snr": "Transit SNR",
    })
    
    formatted_table = formatted_table.round({
        "Period days": 2, "Radius Earths": 2, "Temperature K": 0, "Transit SNR": 1
    })
    formatted_table["Planet likeness"] = formatted_table["Planet likeness"].map(lambda v: f"{v:.1%}")
    
    return formatted_table.to_html(
        index=False, classes="candidate-table", border=0, na_rep="Unknown"
    )


def compile_dashboard(data: pd.DataFrame, metrics: Dict[str, Any], plots: Dict[str, Any], config: PipelineConfig) -> Path:
    """Loads external HTML template, populates variables, and writes finalized page to disk."""
    logger.info(f"Assembling the HTML layout using {config.template_path}...")
    counts = data["koi_disposition"].value_counts()
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    
    if not config.template_path.exists():
        raise FileNotFoundError(f"Missing essential HTML template: {config.template_path}. Please create it first.")
        
    page_template = Template(config.template_path.read_text(encoding="utf-8"))
    table_html = generate_candidates_table(plots["candidates_df"])
    
    page_content = page_template.substitute(
        total_rows=f"{len(data):,}",
        confirmed_rows=f"{counts.get('CONFIRMED', 0):,}",
        candidate_rows=f"{counts.get('CANDIDATE', 0):,}",
        false_positive_rows=f"{counts.get('FALSE POSITIVE', 0):,}",
        population_chart=chart_html(plots["population"], include_library=True),
        score_chart=chart_html(plots["scores"]),
        confusion_chart=chart_html(plots["confusion"]),
        importance_chart=chart_html(plots["importance"]),
        candidate_chart=chart_html(plots["candidate"]),
        candidate_table=table_html,
        generated_at=generated_at,
        leaky_auc=f"{metrics['leaky_auc']:.3f}"
    )
    
    config.output_dir.mkdir(exist_ok=True)
    config.output_path.write_text(page_content, encoding="utf-8")
    logger.info(f"Dashboard successfully built and written to: {config.output_path}")
    return config.output_path
