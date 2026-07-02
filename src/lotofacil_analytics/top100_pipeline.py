from __future__ import annotations

import logging

from .calibrated_weights import load_supervised_calibrated_weights
from .climate_runtime import load_runtime_climate
from .config import AppConfig
from .storage import load_processed_csv
from .top50_refinement import load_top50_refinement_payload
from .top100_engine import Top100BacktestSummary, Top100Summary, build_top100_prediction, run_top100_backtest
from .top100_repair_learning import archive_top100_prediction, load_top100_repair_payload


class Top100Pipeline:
    def __init__(self, *, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def predict(
        self,
        *,
        top_count: int,
        top_pool: int,
        max_overlap: int,
        draw_hour: int,
        draw_minute: int,
        exhaustive_limit: int | None,
        quick_mode: bool = False,
    ) -> Top100Summary:
        concursos = load_processed_csv(self.config.processed_csv_path)
        if concursos.empty:
            raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")
        existing_candidates = load_processed_csv(self.config.optimizer_candidates_csv_path)
        climate_features, target_climate = load_runtime_climate(
            config=self.config,
            concursos=concursos,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
        )
        summary = build_top100_prediction(
            concursos,
            existing_candidates=existing_candidates,
            top_count=top_count,
            top_pool=top_pool,
            max_overlap=max_overlap,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
            exhaustive_limit=exhaustive_limit,
            climate_features=climate_features,
            target_climate=target_climate,
            weights=load_supervised_calibrated_weights(self.config.supervised_calibration_weights_json_path),
            refinement_payload=load_top50_refinement_payload(self.config.top50_refinement_weights_json_path),
            repair_payload=load_top100_repair_payload(self.config.top100_repair_weights_json_path),
            quick_mode=quick_mode,
            prediction_csv_path=self.config.top100_prediction_csv_path,
            report_path=self.config.top100_prediction_report_path,
            excel_path=self.config.top100_prediction_excel_path,
        )
        archive_top100_prediction(self.config, concurso=summary.concurso_alvo)
        self.logger.info("Top 100 salvo em %s", self.config.top100_prediction_csv_path)
        return summary

    def backtest(
        self,
        *,
        n_eval: int,
        min_history: int,
        top_count: int,
        top_pool: int,
        max_overlap: int,
        draw_hour: int,
        draw_minute: int,
        exhaustive_limit: int | None,
    ) -> Top100BacktestSummary:
        concursos = load_processed_csv(self.config.processed_csv_path)
        if concursos.empty:
            raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")
        climate_features, _target_climate = load_runtime_climate(
            config=self.config,
            concursos=concursos,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
        )
        summary = run_top100_backtest(
            concursos,
            climate_features=climate_features,
            n_eval=n_eval,
            min_history=min_history,
            top_count=top_count,
            top_pool=top_pool,
            max_overlap=max_overlap,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
            exhaustive_limit=exhaustive_limit,
            weights=load_supervised_calibrated_weights(self.config.supervised_calibration_weights_json_path),
            refinement_payload=load_top50_refinement_payload(self.config.top50_refinement_weights_json_path),
            results_csv_path=self.config.top100_backtest_csv_path,
            summary_csv_path=self.config.top100_backtest_summary_csv_path,
            excel_path=self.config.top100_backtest_excel_path,
        )
        self.logger.info("Backtest Top 100 salvo em %s", self.config.top100_backtest_csv_path)
        return summary
