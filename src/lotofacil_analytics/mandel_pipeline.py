from __future__ import annotations

import logging

import pandas as pd

from .calibrated_weights import load_calibrated_weights
from .climate_runtime import load_runtime_climate
from .config import AppConfig
from .context_features import build_target_context
from .exhaustive_optimizer import TOTAL_COMBINATIONS, build_exhaustive_candidates
from .mandel_strategy import MandelSummary, run_mandel_strategy
from .storage import load_processed_csv, sanitize_dataframe_for_tabular_output


class MandelPipeline:
    def __init__(self, *, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def _load_or_rebuild_candidates(
        self,
        concursos: pd.DataFrame,
        *,
        draw_hour: int,
        draw_minute: int,
        climate_features: pd.DataFrame,
        target_climate: dict[str, object],
    ) -> pd.DataFrame:
        candidates = load_processed_csv(self.config.optimizer_candidates_csv_path)
        target_context = build_target_context(concursos, draw_hour=draw_hour, draw_minute=draw_minute)
        last_concurso = int(concursos.sort_values("concurso").iloc[-1]["concurso"])
        valid = False
        if (
            not candidates.empty
            and "concurso_base_final" in candidates.columns
            and "contexto_data_proximo_concurso" in candidates.columns
            and "total_combinacoes_avaliadas" in candidates.columns
        ):
            max_evaluated = pd.to_numeric(candidates.get("total_combinacoes_avaliadas"), errors="coerce").max()
            candidate_base = pd.to_numeric(candidates["concurso_base_final"], errors="coerce").max()
            candidate_target_date = str(candidates["contexto_data_proximo_concurso"].iloc[0])
            valid = bool(
                pd.notna(max_evaluated)
                and int(max_evaluated) >= TOTAL_COMBINATIONS
                and pd.notna(candidate_base)
                and int(candidate_base) == last_concurso
                and candidate_target_date == target_context.data_proximo_concurso
                and (climate_features.empty or "score_climatico" in candidates.columns)
            )
        if valid:
            return candidates

        candidates, summary = build_exhaustive_candidates(
            concursos,
            top_games=5000,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
            weights=load_calibrated_weights(self.config.engine_calibration_weights_json_path),
            climate_features=climate_features,
            target_climate=target_climate,
        )
        candidates = sanitize_dataframe_for_tabular_output(candidates)
        summary = sanitize_dataframe_for_tabular_output(summary)
        self.config.optimizer_candidates_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.optimizer_excel_path.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(self.config.optimizer_candidates_csv_path, index=False, encoding="utf-8-sig")
        summary.to_csv(self.config.optimizer_summary_csv_path, index=False, encoding="utf-8-sig")
        with pd.ExcelWriter(self.config.optimizer_excel_path, engine="openpyxl") as writer:
            candidates.to_excel(writer, index=False, sheet_name="candidatos")
            summary.to_excel(writer, index=False, sheet_name="resumo")
        self.logger.info("Candidatos Mandel recalculados em %s", self.config.optimizer_candidates_csv_path)
        return candidates

    def run(
        self,
        *,
        universe_size: int = 18,
        guarantee_hits: int = 14,
        max_reduced_games: int = 80,
        draw_hour: int = 20,
        draw_minute: int = 0,
    ) -> MandelSummary:
        concursos = load_processed_csv(self.config.processed_csv_path)
        if concursos.empty:
            raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")

        climate_features, target_climate = load_runtime_climate(
            config=self.config,
            concursos=concursos,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
        )
        candidates = self._load_or_rebuild_candidates(
            concursos,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
            climate_features=climate_features,
            target_climate=target_climate,
        )
        summary = run_mandel_strategy(
            concursos,
            candidates,
            universe_size=universe_size,
            guarantee_hits=guarantee_hits,
            max_reduced_games=max_reduced_games,
            plan_csv_path=self.config.mandel_plan_csv_path,
            games_csv_path=self.config.mandel_games_csv_path,
            report_path=self.config.mandel_report_path,
            excel_path=self.config.mandel_excel_path,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
        )
        self.logger.info("Plano Mandel salvo em %s", self.config.mandel_plan_csv_path)
        self.logger.info("Jogos Mandel salvos em %s", self.config.mandel_games_csv_path)
        return summary
