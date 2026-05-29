from __future__ import annotations

import logging

import pandas as pd

from .backtest_lotofacil import BacktestSummary, run_backtest
from .config import AppConfig
from .storage import load_processed_csv, sanitize_dataframe_for_tabular_output


class BacktestPipeline:
    def __init__(self, *, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def run(
        self,
        *,
        n_eval: int,
        min_history: int,
        seed: int,
        window: int,
        candidates: int,
    ) -> BacktestSummary:
        concursos = load_processed_csv(self.config.processed_csv_path)
        if concursos.empty:
            raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")

        results, summary = run_backtest(
            concursos,
            n_eval=n_eval,
            min_history=min_history,
            seed=seed,
            window=window,
            candidates=candidates,
        )
        results = sanitize_dataframe_for_tabular_output(results)
        summary = sanitize_dataframe_for_tabular_output(summary)

        self.config.backtest_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.backtest_excel_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(self.config.backtest_csv_path, index=False, encoding="utf-8-sig")
        summary.to_csv(self.config.backtest_summary_csv_path, index=False, encoding="utf-8-sig")

        resumo = pd.DataFrame(
            [
                {"campo": "linhas_resultado", "valor": int(len(results))},
                {"campo": "concursos_avaliados", "valor": int(results["concurso_previsto"].nunique())},
                {"campo": "primeiro_concurso", "valor": int(results["concurso_previsto"].min())},
                {"campo": "ultimo_concurso", "valor": int(results["concurso_previsto"].max())},
                {"campo": "n_eval_solicitado", "valor": int(n_eval)},
                {"campo": "min_history", "valor": int(min_history)},
                {"campo": "seed", "valor": int(seed)},
                {"campo": "window", "valor": int(window)},
                {"campo": "candidates", "valor": int(candidates)},
            ]
        )

        with pd.ExcelWriter(self.config.backtest_excel_path, engine="openpyxl") as writer:
            results.to_excel(writer, index=False, sheet_name="resultados")
            summary.to_excel(writer, index=False, sheet_name="resumo_modelos")
            resumo.to_excel(writer, index=False, sheet_name="resumo_execucao")

        self.logger.info("Backtest salvo em %s", self.config.backtest_csv_path)
        self.logger.info("Resumo do backtest salvo em %s", self.config.backtest_summary_csv_path)
        self.logger.info("Excel do backtest salvo em %s", self.config.backtest_excel_path)

        return BacktestSummary(
            rows=int(len(results)),
            contests=int(results["concurso_previsto"].nunique()),
            first_concurso=int(results["concurso_previsto"].min()),
            last_concurso=int(results["concurso_previsto"].max()),
            results_csv_path=str(self.config.backtest_csv_path),
            summary_csv_path=str(self.config.backtest_summary_csv_path),
            excel_path=str(self.config.backtest_excel_path),
        )
