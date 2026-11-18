
from typing import Dict, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.config import PipelineConfig, logger


def generate_plots(data: pd.DataFrame, metrics: Dict[str, Any], config: PipelineConfig) -> Dict[str, go.Figure]:
    """Generates all interactive Plotly visualization structures."""
    logger.info("Building interactive Plotly visualizations...")
    
    # 1. Main Population Scatter Plot
    plot_data = data[(data["koi_period"] > 0) & (data["koi_prad"] > 0)].copy()
    plot_data["signal_size"] = plot_data["koi_model_snr"].clip(lower=0).fillna(0) + 1
    
    population_fig = px.scatter(
        plot_data, x="koi_period", y="koi_prad", color="koi_disposition",
        size="signal_size", hover_name="kepoi_name",
        hover_data=["koi_duration", "koi_depth", "koi_teq", "koi_model_snr"],
        color_discrete_map=config.disposition_colors, size_max=18,
        log_x=True, log_y=True, opacity=0.72,
        labels={
            "koi_period": "Orbital period in days",
            "koi_prad": "Planetary radius in Earth radii",
            "koi_disposition": "Archive disposition",
        },
        title="The Kepler candidate population", template="plotly_dark", height=590
    )
    
    # 2. Performance Comparison Bar Chart
    model_scores = pd.DataFrame([
        {"model": "Leaky forest", "roc_auc": metrics["leaky_auc"], "kind": "Do not trust"},
        {"model": "Audited forest", "roc_auc": metrics["forest_auc"], "kind": "Audited"},
        {"model": "Logistic baseline", "roc_auc": metrics["logistic_auc"], "kind": "Audited"}
    ])
    score_fig = px.bar(
        model_scores, x="model", y="roc_auc", color="kind", text_auto=".3f",
        color_discrete_map={"Do not trust": "#ff6b9d", "Audited": "#78f7d4"},
        range_y=[0.5, 1.0], labels={"model": "", "roc_auc": "Test ROC-AUC", "kind": "Feature audit"},
        title="Performance before and after the feature audit", template="plotly_dark", height=440
    )
    
    # 3. Confusion Matrix Heatmap
    confusion_fig = px.imshow(
        metrics["confusion_matrix"],
        x=["Predicted false positive", "Predicted confirmed"],
        y=["Actual false positive", "Actual confirmed"],
        labels={"x": "Prediction", "y": "Actual class", "color": "Count"},
        text_auto=True, color_continuous_scale=["#17132f", "#7c3aed", "#78f7d4"],
        title="Audited random-forest confusion matrix", template="plotly_dark", aspect="auto", height=440
    )
    
    # 4. Feature Importances Horizontal Bar
    importance_fig = px.bar(
        metrics["importances"], x="importance", y="feature", orientation="h",
        color="importance", text_auto=".3f", color_continuous_scale=["#6d5dfc", "#d946ef", "#78f7d4"],
        labels={"importance": "Impurity-based importance", "feature": ""},
        title="What the audited forest uses", template="plotly_dark", height=520
    )
    
    # 5. Unresolved Candidates Prediction Map
    candidates = data[data["koi_disposition"] == "CANDIDATE"].copy()
    candidates["signal_size"] = candidates["koi_model_snr"].clip(lower=0).fillna(0) + 1
    candidates["planet_likeness"] = metrics["production_model"].predict_proba(candidates[config.physical_features])[:, 1]
    
    candidate_plot = candidates[(candidates["koi_period"] > 0) & (candidates["koi_prad"] > 0)].copy()
    candidate_fig = px.scatter(
        candidate_plot, x="koi_period", y="koi_prad", color="planet_likeness",
        size="signal_size", hover_name="kepoi_name", hover_data=["koi_teq", "koi_insol", "koi_model_snr"],
        color_continuous_scale=["#362c76", "#a78bfa", "#78f7d4"], range_color=[0, 1],
        size_max=22, log_x=True, log_y=True, opacity=0.82,
        labels={"koi_period": "Orbital period in days", "koi_prad": "Planetary radius in Earth radii", "planet_likeness": "Model probability"},
        title="Unresolved candidates through the audited model", template="plotly_dark", height=560
    )

    return {
        "population": population_fig,
        "scores": score_fig,
        "confusion": confusion_fig,
        "importance": importance_fig,
        "candidate": candidate_fig,
        "candidates_df": candidates
    }
