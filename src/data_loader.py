
from urllib.parse import urlencode
import pandas as pd
from src.config import PipelineConfig, logger


def fetch_data(config: PipelineConfig) -> pd.DataFrame:
    """Retrieves cumulative Kepler records from NASA TAP endpoint."""
    logger.info("Initializing query to NASA Exoplanet Archive...")
    query = f"select {','.join(config.query_columns)} from cumulative"
    request_url = f"{config.tap_endpoint}?{urlencode({'query': query, 'format': 'csv'})}"
    data = pd.read_csv(request_url)
    logger.info(f"Successfully retrieved {len(data):,} astronomical records.")
    return data
