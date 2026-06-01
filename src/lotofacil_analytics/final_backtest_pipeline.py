from __future__ import annotations

import logging

import pandas as pd

from .config import AppConfig
from .final_backtest import FinalBacktestSummary, run_final_score_backtest
from .storage import load_processed_csv, sanitize_dataframe_for_tabular_output


class FinalBacktestPipeline:
    def __init__(self, *, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def run(
        self,
        *,
        n_eval: int,
        min_history: int,
        seed: int,
        candidate_pool: int,
        top_games: int,
        generations: int,
        population: int,
        max_overlap: int,
        draw_hour: int,
        draw_minute: int,
    ) -> FinalBacktestSummary:
        concursos = load_processed_csv(self.config.processed_csv_path)
        if concursos.empty:
            raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")

        results, summary = run_final_score_backtest(
            concursos,
            n_eval=n_eval,
            min_history=min_history,
            seed=seed,
            candidate_pool=candidate_pool,
            top_games=top_games,
            generations=generations,
            population=population,
            max_overlap=max_overlap,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
        )
        results = sanitize_dataframe_for_tabular_output(results)
        summary = sanitize_dataframe_for_tabular_output(summary)

        self.config.final_backtest_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.final_backtest_excel_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(self.config.final_backtest_csv_path, index=False, encoding="utf-8-sig")
        summary.to_csv(self.config.final_backtest_summary_csv_path, index=False, encoding="utf-8-sig")

        with pd.ExcelWriter(self.config.final_backtest_excel_path, engine="openpyxl") as writer:
            results.to_excel(writer, index=False, sheet_name="resultados")
            summary.to_excel(writer, index=False, sheet_name="resumo")

        self.logger.info("Backtest score final salvo em %s", self.config.final_backtest_csv_path)
        self.logger.info("Resumo do backtest score final salvo em %s", self.config.final_backtest_summary_csv_path)
        self.logger.info("Excel do backtest score final salvo em %s", self.config.final_backtest_excel_path)

        return FinalBacktestSummary(
            rows=int(len(results)),
            contests=int(results["concurso_previsto"].nunique()),
            first_concurso=int(results["concurso_previsto"].min()),
            last_concurso=int(results["concurso_previsto"].max()),
            results_csv_path=str(self.config.final_backtest_csv_path),
            summary_csv_path=str(self.config.final_backtest_summary_csv_path),
            excel_path=str(self.config.final_backtest_excel_path),
        )
