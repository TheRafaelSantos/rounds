from __future__ import annotations

import logging

from .config import AppConfig
from .features_base import FeatureSummary, build_base_features
from .storage import load_processed_csv, save_processed_outputs


class FeaturePipeline:
    def __init__(self, *, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def build_base_features(self) -> FeatureSummary:
        concursos = load_processed_csv(self.config.processed_csv_path)
        if concursos.empty:
            raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")

        features = build_base_features(concursos)
        save_processed_outputs(
            features,
            csv_path=self.config.features_base_csv_path,
            excel_path=self.config.features_base_excel_path,
        )
        self.logger.info("Features base salvas em %s", self.config.features_base_csv_path)
        self.logger.info("Excel de features salvo em %s", self.config.features_base_excel_path)

        return FeatureSummary(
            total_rows=int(len(features)),
            first_concurso=int(features["concurso"].min()),
            last_concurso=int(features["concurso"].max()),
            csv_path=str(self.config.features_base_csv_path),
            excel_path=str(self.config.features_base_excel_path),
        )
