from __future__ import annotations

import logging

import pandas as pd

from .calibration_lab import CalibrationLabSummary, run_calibration_lab
from .config import AppConfig
from .storage import load_processed_csv


class CalibrationLabPipeline:
    def __init__(self, *, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def run(
        self,
        *,
        from_concurso: int,
        to_concurso: int | None,
        max_attempts: int,
        top_games: int,
        exhaustive_limit: int | None,
        max_overlap: int,
        seed: int,
        draw_hour: int,
        draw_minute: int,
        min_history: int,
        max_runtime_seconds: int,
        reset: bool,
    ) -> CalibrationLabSummary:
        concursos = load_processed_csv(self.config.processed_csv_path)
        if concursos.empty:
            raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")
        climate_features = (
            pd.read_csv(self.config.climate_csv_path, encoding="utf-8-sig")
            if self.config.climate_csv_path.exists()
            else pd.DataFrame()
        )
        summary = run_calibration_lab(
            concursos,
            climate_features=climate_features,
            from_concurso=from_concurso,
            to_concurso=to_concurso,
            max_attempts=max_attempts,
            top_games=top_games,
            exhaustive_limit=exhaustive_limit,
            max_overlap=max_overlap,
            seed=seed,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
            min_history=min_history,
            max_runtime_seconds=max_runtime_seconds,
            reset=reset,
            state_json_path=self.config.calibration_lab_state_json_path,
            attempts_csv_path=self.config.calibration_lab_attempts_csv_path,
            winners_csv_path=self.config.calibration_lab_winners_csv_path,
            elites_csv_path=self.config.calibration_lab_elites_csv_path,
            summary_csv_path=self.config.calibration_lab_summary_csv_path,
            average_weights_csv_path=self.config.calibration_lab_average_weights_csv_path,
            excel_path=self.config.calibration_lab_excel_path,
            engine_weights_json_path=self.config.engine_calibration_weights_json_path,
            cache_dir=self.config.calibration_lab_cache_dir,
        )
        self.logger.info("Calibracao 24/7 estado salvo em %s", self.config.calibration_lab_state_json_path)
        return summary
