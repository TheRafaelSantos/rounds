from __future__ import annotations

import logging

import pandas as pd

from .combinacoes import CombinacoesSummary, build_combinacoes_outputs
from .config import AppConfig
from .storage import load_processed_csv, sanitize_dataframe_for_tabular_output


class CombinacoesPipeline:
    def __init__(self, *, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def build_combinacoes(self) -> CombinacoesSummary:
        concursos = load_processed_csv(self.config.processed_csv_path)
        if concursos.empty:
            raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")

        features, pares, trios, quartetos = build_combinacoes_outputs(concursos)
        features = sanitize_dataframe_for_tabular_output(features)
        pares = sanitize_dataframe_for_tabular_output(pares)
        trios = sanitize_dataframe_for_tabular_output(trios)
        quartetos = sanitize_dataframe_for_tabular_output(quartetos)

        self.config.combinacoes_features_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.combinacoes_excel_path.parent.mkdir(parents=True, exist_ok=True)

        features.to_csv(self.config.combinacoes_features_csv_path, index=False, encoding="utf-8-sig")
        pares.to_csv(self.config.combinacoes_pares_csv_path, index=False, encoding="utf-8-sig")
        trios.to_csv(self.config.combinacoes_trios_csv_path, index=False, encoding="utf-8-sig")
        quartetos.to_csv(self.config.combinacoes_quartetos_csv_path, index=False, encoding="utf-8-sig")

        resumo = pd.DataFrame(
            [
                {"campo": "concursos", "valor": int(concursos["concurso"].nunique())},
                {"campo": "features_rows", "valor": int(len(features))},
                {"campo": "pares_rows", "valor": int(len(pares))},
                {"campo": "trios_rows", "valor": int(len(trios))},
                {"campo": "quartetos_rows", "valor": int(len(quartetos))},
            ]
        )

        with pd.ExcelWriter(self.config.combinacoes_excel_path, engine="openpyxl") as writer:
            features.to_excel(writer, index=False, sheet_name="features")
            pares.to_excel(writer, index=False, sheet_name="pares")
            trios.to_excel(writer, index=False, sheet_name="trios")
            quartetos.to_excel(writer, index=False, sheet_name="quartetos")
            resumo.to_excel(writer, index=False, sheet_name="resumo")

        self.logger.info("Features combinatorias salvas em %s", self.config.combinacoes_features_csv_path)
        self.logger.info("Pares salvos em %s", self.config.combinacoes_pares_csv_path)
        self.logger.info("Trios salvos em %s", self.config.combinacoes_trios_csv_path)
        self.logger.info("Quartetos salvos em %s", self.config.combinacoes_quartetos_csv_path)
        self.logger.info("Excel combinatorio salvo em %s", self.config.combinacoes_excel_path)

        return CombinacoesSummary(
            concursos=int(concursos["concurso"].nunique()),
            features_rows=int(len(features)),
            pares_rows=int(len(pares)),
            trios_rows=int(len(trios)),
            quartetos_rows=int(len(quartetos)),
            first_concurso=int(concursos["concurso"].min()),
            last_concurso=int(concursos["concurso"].max()),
            features_csv_path=str(self.config.combinacoes_features_csv_path),
            pares_csv_path=str(self.config.combinacoes_pares_csv_path),
            trios_csv_path=str(self.config.combinacoes_trios_csv_path),
            quartetos_csv_path=str(self.config.combinacoes_quartetos_csv_path),
            excel_path=str(self.config.combinacoes_excel_path),
        )
