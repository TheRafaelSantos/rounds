from __future__ import annotations

import logging

import pandas as pd

from .config import AppConfig
from .dezenas_history import DezenasSummary, build_dezenas_outputs
from .storage import load_processed_csv, sanitize_dataframe_for_tabular_output


class DezenasPipeline:
    def __init__(self, *, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def build_history(self) -> DezenasSummary:
        concursos = load_processed_csv(self.config.processed_csv_path)
        if concursos.empty:
            raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")

        dezenas_long, dezenas_historico = build_dezenas_outputs(concursos)
        dezenas_long = sanitize_dataframe_for_tabular_output(dezenas_long)
        dezenas_historico = sanitize_dataframe_for_tabular_output(dezenas_historico)

        self.config.dezenas_long_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.dezenas_historico_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.dezenas_historico_excel_path.parent.mkdir(parents=True, exist_ok=True)

        dezenas_long.to_csv(self.config.dezenas_long_csv_path, index=False, encoding="utf-8-sig")
        dezenas_historico.to_csv(self.config.dezenas_historico_csv_path, index=False, encoding="utf-8-sig")

        resumo = pd.DataFrame(
            [
                {"campo": "concursos", "valor": int(concursos["concurso"].nunique())},
                {"campo": "linhas_dezenas_long", "valor": int(len(dezenas_long))},
                {"campo": "linhas_dezenas_historico", "valor": int(len(dezenas_historico))},
                {"campo": "primeiro_concurso", "valor": int(concursos["concurso"].min())},
                {"campo": "ultimo_concurso", "valor": int(concursos["concurso"].max())},
            ]
        )

        with pd.ExcelWriter(self.config.dezenas_historico_excel_path, engine="openpyxl") as writer:
            dezenas_long.to_excel(writer, index=False, sheet_name="dezenas_long")
            dezenas_historico.to_excel(writer, index=False, sheet_name="dezenas_historico")
            resumo.to_excel(writer, index=False, sheet_name="resumo")

        self.logger.info("Dezenas long salvas em %s", self.config.dezenas_long_csv_path)
        self.logger.info("Historico por dezena salvo em %s", self.config.dezenas_historico_csv_path)
        self.logger.info("Excel de historico por dezena salvo em %s", self.config.dezenas_historico_excel_path)

        return DezenasSummary(
            concursos=int(concursos["concurso"].nunique()),
            long_rows=int(len(dezenas_long)),
            historico_rows=int(len(dezenas_historico)),
            first_concurso=int(concursos["concurso"].min()),
            last_concurso=int(concursos["concurso"].max()),
            long_csv_path=str(self.config.dezenas_long_csv_path),
            historico_csv_path=str(self.config.dezenas_historico_csv_path),
            excel_path=str(self.config.dezenas_historico_excel_path),
        )
