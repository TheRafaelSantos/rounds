from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from .climate_features import climate_lookup_for_target, load_climate_features, neutral_target_climate
from .config import AppConfig


def load_runtime_climate(
    *,
    config: AppConfig,
    concursos: pd.DataFrame,
    draw_hour: int,
    draw_minute: int,
    force: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    climate_features = load_climate_features(config.climate_csv_path)
    if climate_features.empty:
        return climate_features, neutral_target_climate("base_climatica_indisponivel")
    target_climate = climate_lookup_for_target(
        concursos,
        climate_features,
        cache_dir=config.climate_cache_dir,
        draw_hour=draw_hour,
        draw_minute=draw_minute,
        timeout_seconds=config.timeout_seconds,
        retries=config.max_retries,
        sleep_seconds=config.request_sleep_seconds,
        force=force,
    )
    return climate_features, target_climate
