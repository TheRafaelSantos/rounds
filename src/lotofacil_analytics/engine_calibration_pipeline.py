from __future__ import annotations

import logging

import pandas as pd

from .climate_features import load_climate_features
from .config import AppConfig
from .engine_calibration import EngineCalibrationSummary, run_engine_calibration
from .storage import load_processed_csv, sanitize_dataframe_for_tabular_output


class EngineCalibrationPipeline:
    def __init__(self, *, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def run(
        self,
        *,
        from_concurso: int = 2500,
        to_concurso: int | None = None,
        baseline_samples: int = 30,
        seed: int = 123,
        draw_hour: int = 20,
        draw_minute: int = 0,
    ) -> EngineCalibrationSummary:
        concursos = load_processed_csv(self.config.processed_csv_path)
        if concursos.empty:
            raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")
        climate_features = load_climate_features(self.config.climate_csv_path)
        results, summary, _payload = run_engine_calibration(
            concursos,
            climate_features=climate_features,
            from_concurso=from_concurso,
            to_concurso=to_concurso,
            baseline_samples=baseline_samples,
            seed=seed,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
            weights_json_path=self.config.engine_calibration_weights_json_path,
        )
        results = sanitize_dataframe_for_tabular_output(results)
        summary = sanitize_dataframe_for_tabular_output(summary)
        self.config.engine_calibration_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.engine_calibration_excel_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(self.config.engine_calibration_csv_path, index=False, encoding="utf-8-sig")
        summary.to_csv(self.config.engine_calibration_summary_csv_path, index=False, encoding="utf-8-sig")
        with pd.ExcelWriter(self.config.engine_calibration_excel_path, engine="openpyxl") as writer:
            results.to_excel(writer, index=False, sheet_name="calibracao")
            summary.to_excel(writer, index=False, sheet_name="resumo")

        self.logger.info("Calibracao do motor salva em %s", self.config.engine_calibration_csv_path)
        return EngineCalibrationSummary(
            rows=int(len(results)),
            contests=int(results["concurso"].nunique()),
            first_concurso=int(results["concurso"].min()),
            last_concurso=int(results["concurso"].max()),
            weights_json_path=str(self.config.engine_calibration_weights_json_path),
            results_csv_path=str(self.config.engine_calibration_csv_path),
            summary_csv_path=str(self.config.engine_calibration_summary_csv_path),
            excel_path=str(self.config.engine_calibration_excel_path),
        )
