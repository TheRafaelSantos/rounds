from __future__ import annotations

import logging

from .climate_features import load_climate_features
from .config import AppConfig
from .storage import load_processed_csv
from .supervised_calibration import SupervisedCalibrationSummary, run_supervised_calibration


class SupervisedCalibrationPipeline:
    def __init__(self, *, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def run(
        self,
        *,
        from_concurso: int,
        to_concurso: int | None,
        samples: int,
        max_contests: int,
        seed: int,
        draw_hour: int,
        draw_minute: int,
        min_history: int,
        reset: bool,
    ) -> SupervisedCalibrationSummary:
        concursos = load_processed_csv(self.config.processed_csv_path)
        if concursos.empty:
            raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")
        climate_features = load_climate_features(self.config.climate_csv_path)
        summary = run_supervised_calibration(
            concursos,
            climate_features=climate_features,
            from_concurso=from_concurso,
            to_concurso=to_concurso,
            samples=samples,
            max_contests=max_contests,
            seed=seed,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
            min_history=min_history,
            reset=reset,
            state_json_path=self.config.supervised_calibration_state_json_path,
            results_csv_path=self.config.supervised_calibration_results_csv_path,
            summary_csv_path=self.config.supervised_calibration_summary_csv_path,
            weights_csv_path=self.config.supervised_calibration_weights_csv_path,
            excel_path=self.config.supervised_calibration_excel_path,
            weights_json_path=self.config.supervised_calibration_weights_json_path,
        )
        self.logger.info(
            "Calibracao supervisionada salva em %s",
            self.config.supervised_calibration_state_json_path,
        )
        return summary
