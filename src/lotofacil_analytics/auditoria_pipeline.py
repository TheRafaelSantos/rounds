from __future__ import annotations

import logging

import pandas as pd

from .auditoria import AuditoriaSummary, build_auditoria
from .config import AppConfig
from .storage import load_processed_csv, sanitize_dataframe_for_tabular_output


class AuditoriaPipeline:
    def __init__(self, *, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def run(self, *, monte_carlo_runs: int, seed: int) -> AuditoriaSummary:
        concursos = load_processed_csv(self.config.processed_csv_path)
        if concursos.empty:
            raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")

        resumo, dezenas, anomalias, monte_carlo = build_auditoria(
            concursos,
            monte_carlo_runs=monte_carlo_runs,
            seed=seed,
        )
        resumo = sanitize_dataframe_for_tabular_output(resumo)
        dezenas = sanitize_dataframe_for_tabular_output(dezenas)
        anomalias = sanitize_dataframe_for_tabular_output(anomalias)
        monte_carlo = sanitize_dataframe_for_tabular_output(monte_carlo)

        self.config.auditoria_resumo_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.auditoria_excel_path.parent.mkdir(parents=True, exist_ok=True)

        resumo.to_csv(self.config.auditoria_resumo_csv_path, index=False, encoding="utf-8-sig")
        dezenas.to_csv(self.config.auditoria_dezenas_csv_path, index=False, encoding="utf-8-sig")
        anomalias.to_csv(self.config.auditoria_anomalias_csv_path, index=False, encoding="utf-8-sig")
        monte_carlo.to_csv(self.config.auditoria_monte_carlo_csv_path, index=False, encoding="utf-8-sig")

        with pd.ExcelWriter(self.config.auditoria_excel_path, engine="openpyxl") as writer:
            resumo.to_excel(writer, index=False, sheet_name="resumo")
            dezenas.to_excel(writer, index=False, sheet_name="dezenas")
            anomalias.to_excel(writer, index=False, sheet_name="anomalias")
            monte_carlo.to_excel(writer, index=False, sheet_name="monte_carlo")

        self.logger.info("Auditoria resumo salva em %s", self.config.auditoria_resumo_csv_path)
        self.logger.info("Auditoria dezenas salva em %s", self.config.auditoria_dezenas_csv_path)
        self.logger.info("Auditoria anomalias salva em %s", self.config.auditoria_anomalias_csv_path)
        self.logger.info("Auditoria Monte Carlo salva em %s", self.config.auditoria_monte_carlo_csv_path)
        self.logger.info("Excel de auditoria salvo em %s", self.config.auditoria_excel_path)

        return AuditoriaSummary(
            concursos=int(concursos["concurso"].nunique()),
            resumo_rows=int(len(resumo)),
            dezenas_rows=int(len(dezenas)),
            anomalias_rows=int(len(anomalias)),
            monte_carlo_rows=int(len(monte_carlo)),
            resumo_csv_path=str(self.config.auditoria_resumo_csv_path),
            dezenas_csv_path=str(self.config.auditoria_dezenas_csv_path),
            anomalias_csv_path=str(self.config.auditoria_anomalias_csv_path),
            monte_carlo_csv_path=str(self.config.auditoria_monte_carlo_csv_path),
            excel_path=str(self.config.auditoria_excel_path),
        )
