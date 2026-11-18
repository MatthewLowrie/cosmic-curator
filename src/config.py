
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import List, Dict

# Setup clean centralized logging with UTC timestamps
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s UTC [%(levelname)s] %(message)s"
)
logger = logging.getLogger("CosmicCurator")


@dataclass(frozen=True)
class PipelineConfig:
    """Read-only configuration for features, targets, models, and file paths."""
    random_state: int = 42
    tap_endpoint: str = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    template_path: Path = field(default_factory=lambda: Path("template.html"))
    output_dir: Path = field(default_factory=lambda: Path("docs"))
    output_path: Path = field(default_factory=lambda: Path("docs/index.html"))
    
    physical_features: List[str] = field(default_factory=lambda: [
        "koi_period", "koi_impact", "koi_duration", "koi_depth", "koi_prad",
        "koi_teq", "koi_insol", "koi_model_snr", "koi_steff", "koi_slogg",
        "koi_srad", "koi_kepmag"
    ])
    
    leakage_features: List[str] = field(default_factory=lambda: [
        "koi_score", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"
    ])

    feature_labels: Dict[str, str] = field(default_factory=lambda: {
        "koi_period": "Orbital period",
        "koi_impact": "Impact parameter",
        "koi_duration": "Transit duration",
        "koi_depth": "Transit depth",
        "koi_prad": "Planetary radius",
        "koi_teq": "Equilibrium temperature",
        "koi_insol": "Insolation flux",
        "koi_model_snr": "Transit signal-to-noise",
        "koi_steff": "Stellar temperature",
        "koi_slogg": "Stellar surface gravity",
        "koi_srad": "Stellar radius",
        "koi_kepmag": "Kepler-band magnitude",
    })

    disposition_colors: Dict[str, str] = field(default_factory=lambda: {
        "CONFIRMED": "#78f7d4",
        "CANDIDATE": "#a78bfa",
        "FALSE POSITIVE": "#ff6b9d",
    })

    @property
    def query_columns(self) -> List[str]:
        return ["kepoi_name", "koi_disposition"] + self.physical_features + self.leakage_features
