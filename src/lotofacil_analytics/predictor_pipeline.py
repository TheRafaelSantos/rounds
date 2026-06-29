from __future__ import annotations

import logging

import pandas as pd

from .calibrated_weights import load_supervised_calibrated_weights
from .climate_runtime import load_runtime_climate
from .config import AppConfig
from .predictor import PredictionSummary, build_final_prediction
from .storage import load_processed_csv


class PredictorPipeline:
    def __init__(self, *, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def predict(
        self,
        *,
        seed: int,
        candidate_pool: int,
        top_games: int,
        generations: int,
        population: int,
        max_overlap: int,
        draw_hour: int = 20,
        draw_minute: int = 0,
        engine: str = "exaustivo",
        exhaustive_limit: int | None = None,
    ) -> PredictionSummary:
        concursos = load_processed_csv(self.config.processed_csv_path)
        if concursos.empty:
            raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")

        candidates = load_processed_csv(self.config.optimizer_candidates_csv_path)
        climate_features, target_climate = load_runtime_climate(
            config=self.config,
            concursos=concursos,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
        )
        summary = build_final_prediction(
            concursos,
            existing_candidates=candidates,
            seed=seed,
            candidate_pool=candidate_pool,
            top_games=top_games,
            generations=generations,
            population=population,
            max_overlap=max_overlap,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
            prediction_csv_path=self.config.prediction_csv_path,
            report_path=self.config.prediction_report_path,
            excel_path=self.config.prediction_excel_path,
            engine=engine,
            exhaustive_limit=exhaustive_limit,
            climate_features=climate_features,
            target_climate=target_climate,
            weights=load_supervised_calibrated_weights(
                self.config.supervised_calibration_weights_json_path,
                fallback_path=self.config.engine_calibration_weights_json_path,
            ),
        )

        self.logger.info("Predicao final salva em %s", self.config.prediction_csv_path)
        self.logger.info("Relatorio tecnico salvo em %s", self.config.prediction_report_path)
        self.logger.info("Excel da predicao salvo em %s", self.config.prediction_excel_path)
        return summary
