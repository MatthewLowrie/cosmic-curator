
from pathlib import Path

import plotly.express as px
from plotly.io import write_html

# 1. Import configurations and functions from our new src/ package folder
from src.config import PipelineConfig
from src.data_loader import fetch_data
from src.model_engine import make_forest, make_logistic


def main() -> None:
    # 2. Instantiate our read-only pipeline configuration
    config = PipelineConfig()
    
    # 3. Use the modular loader with our configuration
    data = fetch_data(config)
    
    labeled = data[
        data["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])
    ].copy()
    labeled["target"] = (labeled["koi_disposition"] == "CONFIRMED").astype(int)
    candidates = data[data["koi_disposition"] == "CANDIDATE"].copy()

    # 4. Instantiate our modular models, passing config where required
    forest = make_forest(config)
    logistic = make_logistic()
    
    # 5. Fit the models using our config-defined physical features
    forest.fit(labeled[config.physical_features], labeled["target"])
    logistic.fit(labeled[config.physical_features], labeled["target"])

    # 6. Predict probabilities and calculate absolute disagreement gaps
    candidates["forest_probability"] = forest.predict_proba(
        candidates[config.physical_features]
    )[:, 1]
    candidates["logistic_probability"] = logistic.predict_proba(
        candidates[config.physical_features]
    )[:, 1]
    
    candidates["disagreement"] = abs(
        candidates["forest_probability"] - candidates["logistic_probability"]
    )
    candidates["signal_size"] = (
        candidates["koi_model_snr"].clip(lower=0).fillna(0) + 1
    )

    # 7. Render our interactive scatter chart
    figure = px.scatter(
        candidates,
        x="logistic_probability",
        y="forest_probability",
        color="disagreement",
        size="signal_size",
        hover_name="kepoi_name",
        hover_data=["koi_period", "koi_prad", "koi_teq", "koi_model_snr"],
        color_continuous_scale=["#312e81", "#d946ef", "#78f7d4"],
        range_color=[0, 1],
        range_x=[0, 1],
        range_y=[0, 1],
        size_max=24,
        opacity=0.82,
        labels={
            "logistic_probability": "Logistic probability",
            "forest_probability": "Random-forest probability",
            "disagreement": "Absolute gap",
        },
        title="Cosmic Curator: where the audited models disagree",
        template="plotly_dark",
        height=720,
    )

    # 8. Ensure output folder exists and compile the standalone page
    config.output_dir.mkdir(exist_ok=True)
    output = config.output_dir / "model-disagreement.html"
    
    write_html(
        figure,
        output,
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
        config={"displaylogo": False, "responsive": True},
    )

    # 9. Sort and report the top 10 most contested candidates
    strongest = candidates.sort_values("disagreement", ascending=False).head(10)
    print(f"\nBuilt {output}")
    print(
        strongest[
            [
                "kepoi_name",
                "logistic_probability",
                "forest_probability",
                "disagreement",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
