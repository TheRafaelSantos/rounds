from __future__ import annotations

import logging

import pandas as pd

from .config import AppConfig
from .optimizer import OptimizerSummary, build_optimized_candidates
from .storage import load_processed_csv, sanitize_dataframe_for_tabular_output


class OptimizerPipeline:
    def __init__(self, *, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def run(
        self,
        *,
        seed: int,
        candidate_pool: int,
        top_games: int,
        generations: int,
        population: int,
    ) -> OptimizerSummary:
        concursos = load_processed_csv(self.config.processed_csv_path)
        if concursos.empty:
            raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")

        candidates, summary = build_optimized_candidates(
            concursos,
            seed=seed,
            candidate_pool=candidate_pool,
            top_games=top_games,
            generations=generations,
            population=population,
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

        self.logger.info("Candidatos otimizados salvos em %s", self.config.optimizer_candidates_csv_path)
        self.logger.info("Resumo do otimizador salvo em %s", self.config.optimizer_summary_csv_path)
        self.logger.info("Excel do otimizador salvo em %s", self.config.optimizer_excel_path)

        return OptimizerSummary(
            candidates_rows=int(len(candidates)),
            summary_rows=int(len(summary)),
            candidates_csv_path=str(self.config.optimizer_candidates_csv_path),
            summary_csv_path=str(self.config.optimizer_summary_csv_path),
            excel_path=str(self.config.optimizer_excel_path),
        )
