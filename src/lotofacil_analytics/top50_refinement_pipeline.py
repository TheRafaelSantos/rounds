from __future__ import annotations

import logging

from .calibrated_weights import load_supervised_calibrated_weights
from .climate_runtime import load_runtime_climate
from .config import AppConfig
from .storage import load_processed_csv
from .top50_refinement import load_top50_refinement_payload
from .top50_refinement_engine import (
    Top50RefinementSummary,
    load_top50_refinement_status,
    run_top50_refinement,
)


class Top50RefinementPipeline:
    def __init__(self, *, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def run(
        self,
        *,
        from_concurso: int,
        to_concurso: int | None,
        max_contests: int,
        min_history: int,
        top_pool: int,
        exhaustive_limit: int | None,
        seed: int,
        draw_hour: int,
        draw_minute: int,
        reset: bool,
    ) -> Top50RefinementSummary:
        concursos = load_processed_csv(self.config.processed_csv_path)
        if concursos.empty:
            raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")
        climate_features, _target_climate = load_runtime_climate(
            config=self.config,
            concursos=concursos,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
        )
        summary = run_top50_refinement(
            concursos,
            climate_features=climate_features,
            from_concurso=from_concurso,
            to_concurso=to_concurso,
            max_contests=max_contests,
            min_history=min_history,
            top_pool=top_pool,
            exhaustive_limit=exhaustive_limit,
            seed=seed,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
            reset=reset,
            base_weights=load_supervised_calibrated_weights(self.config.supervised_calibration_weights_json_path),
            state_json_path=self.config.top50_refinement_state_json_path,
            results_csv_path=self.config.top50_refinement_results_csv_path,
            summary_csv_path=self.config.top50_refinement_summary_csv_path,
            weights_csv_path=self.config.top50_refinement_weights_csv_path,
            excel_path=self.config.top50_refinement_excel_path,
            weights_json_path=self.config.top50_refinement_weights_json_path,
        )
        self.logger.info("Refinador Top50 salvo em %s", self.config.top50_refinement_weights_json_path)
        return summary

    def status(self) -> dict:
        return load_top50_refinement_status(
            state_json_path=self.config.top50_refinement_state_json_path,
            results_csv_path=self.config.top50_refinement_results_csv_path,
            summary_csv_path=self.config.top50_refinement_summary_csv_path,
            weights_csv_path=self.config.top50_refinement_weights_csv_path,
            weights_json_path=self.config.top50_refinement_weights_json_path,
        )

    def payload(self) -> dict | None:
        return load_top50_refinement_payload(self.config.top50_refinement_weights_json_path)
