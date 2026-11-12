from pathlib import Path
from string import Template
from urllib.parse import urlencode

import pandas as pd
import plotly.express as px
from plotly.io import to_html


RANDOM_STATE = 42
TAP_ENDPOINT = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

PHYSICAL_FEATURES = [
    "koi_period",
    "koi_impact",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_teq",
    "koi_insol",
    "koi_model_snr",
    "koi_steff",
    "koi_slogg",
    "koi_srad",
    "koi_kepmag",
]

LEAKAGE_FEATURES = [
    "koi_score",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
]

QUERY_COLUMNS = ["kepoi_name", "koi_disposition", *PHYSICAL_FEATURES, *LEAKAGE_FEATURES]

FEATURE_LABELS = {
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
}

DISPOSITION_COLORS = {
    "CONFIRMED": "#78f7d4",
    "CANDIDATE": "#a78bfa",
    "FALSE POSITIVE": "#ff6b9d",
}


def fetch_data():
    query = f"select {', '.join(QUERY_COLUMNS)} from cumulative"
    request_url = f"{TAP_ENDPOINT}?{urlencode({'query': query, 'format': 'csv'})}"
    return pd.read_csv(request_url)

def chart_html(figure, include_library=False):
    return to_html(
        figure,
        full_html=False,
        include_plotlyjs=include_library,
        config={
            "displaylogo" : False,
            "responsive" : True,
        },
    )