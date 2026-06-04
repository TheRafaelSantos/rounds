from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .calibration_pilot import CalibrationPilotSummary, run_calibration_pilot
from .config import AppConfig
from .storage import load_processed_csv


class CalibrationPilotPipeline:
    def __init__(self, *, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def _paths_for_concurso(self, concurso: int) -> dict[str, Path]:
        suffix = str(int(concurso))
        return {
            "candidates_csv_path": self.config.processed_dir / f"lotofacil_calibration_pilot_candidates_{suffix}.csv",
            "results_csv_path": self.config.processed_dir / f"lotofacil_calibration_pilot_results_{suffix}.csv",
            "summary_csv_path": self.config.processed_dir / f"lotofacil_calibration_pilot_summary_{suffix}.csv",
            "state_json_path": self.config.processed_dir / f"lotofacil_calibration_pilot_state_{suffix}.json",
            "excel_path": self.config.exports_dir / f"lotofacil_calibration_pilot_{suffix}.xlsx",
        }

    def run(
        self,
        *,
        target_concurso: int,
        attempts: int,
        candidate_pool: int,
        exhaustive_limit: int | None,
        max_overlap: int,
        seed: int,
        draw_hour: int,
        draw_minute: int,
        reset: bool = False,
    ) -> CalibrationPilotSummary:
        concursos = load_processed_csv(self.config.processed_csv_path)
        if concursos.empty:
            raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")
        climate_features = (
            pd.read_csv(self.config.climate_csv_path)
            if self.config.climate_csv_path.exists()
            else pd.DataFrame()
        )
        paths = self._paths_for_concurso(target_concurso)
        summary = run_calibration_pilot(
            concursos,
            climate_features=climate_features,
            target_concurso=target_concurso,
            attempts=attempts,
            candidate_pool=candidate_pool,
            exhaustive_limit=exhaustive_limit,
            max_overlap=max_overlap,
            seed=seed,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
            reset=reset,
            **paths,
        )
        self.logger.info("Piloto de calibracao salvo em %s", paths["results_csv_path"])
        return summary
