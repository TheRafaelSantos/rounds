from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from .config import AppConfig
from .storage import load_processed_csv, sanitize_dataframe_for_tabular_output
from .transition_analysis import build_transition_outputs


@dataclass(frozen=True)
class TransitionSummary:
    transition_rows: int
    summary_rows: int
    number_rows: int
    csv_path: str
    summary_csv_path: str
    number_csv_path: str
    excel_path: str

    def to_console(self) -> str:
        return "\n".join(
            [
                "",
                "Resumo Lotofacil Analytics - Transicoes",
                f"Transicoes analisadas: {self.transition_rows}",
                f"Linhas de resumo: {self.summary_rows}",
                f"Linhas por dezena: {self.number_rows}",
                f"CSV transicoes: {self.csv_path}",
                f"CSV resumo: {self.summary_csv_path}",
                f"CSV dezenas: {self.number_csv_path}",
                f"Excel: {self.excel_path}",
                "Mensagem: Concurso N comparado com N+1 para alimentar insights e score de transicao.",
            ]
        )


class TransitionPipeline:
    def __init__(self, *, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def run(self) -> TransitionSummary:
        concursos = load_processed_csv(self.config.processed_csv_path)
        if concursos.empty:
            raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")

        transitions, summary, number_stats = build_transition_outputs(concursos)
        transitions = sanitize_dataframe_for_tabular_output(transitions)
        summary = sanitize_dataframe_for_tabular_output(summary)
        number_stats = sanitize_dataframe_for_tabular_output(number_stats)

        self.config.transition_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.transition_excel_path.parent.mkdir(parents=True, exist_ok=True)
        transitions.to_csv(self.config.transition_csv_path, index=False, encoding="utf-8-sig")
        summary.to_csv(self.config.transition_summary_csv_path, index=False, encoding="utf-8-sig")
        number_stats.to_csv(self.config.transition_dezenas_csv_path, index=False, encoding="utf-8-sig")

        with pd.ExcelWriter(self.config.transition_excel_path, engine="openpyxl") as writer:
            transitions.to_excel(writer, index=False, sheet_name="transicoes")
            summary.to_excel(writer, index=False, sheet_name="resumo")
            number_stats.to_excel(writer, index=False, sheet_name="dezenas")

        self.logger.info("Transicoes salvas em %s", self.config.transition_csv_path)
        self.logger.info("Resumo de transicoes salvo em %s", self.config.transition_summary_csv_path)
        self.logger.info("Dezenas de transicao salvas em %s", self.config.transition_dezenas_csv_path)
        return TransitionSummary(
            transition_rows=int(len(transitions)),
            summary_rows=int(len(summary)),
            number_rows=int(len(number_stats)),
            csv_path=str(self.config.transition_csv_path),
            summary_csv_path=str(self.config.transition_summary_csv_path),
            number_csv_path=str(self.config.transition_dezenas_csv_path),
            excel_path=str(self.config.transition_excel_path),
        )
