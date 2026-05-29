from __future__ import annotations

import logging

import pandas as pd

from .config import AppConfig
from .ml_temporal import MLSummary, run_ml_temporal
from .storage import load_processed_csv, sanitize_dataframe_for_tabular_output


class MLPipeline:
    def __init__(self, *, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def run(
        self,
        *,
        train_ratio: float,
        validation_ratio: float,
        epochs: int,
        learning_rate: float,
        l2: float,
        seed: int,
    ) -> MLSummary:
        concursos = load_processed_csv(self.config.processed_csv_path)
        if concursos.empty:
            raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")

        dataset, predictions, summary = run_ml_temporal(
            concursos,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            seed=seed,
        )
        dataset = sanitize_dataframe_for_tabular_output(dataset)
        predictions = sanitize_dataframe_for_tabular_output(predictions)
        summary = sanitize_dataframe_for_tabular_output(summary)

        self.config.ml_dataset_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.ml_excel_path.parent.mkdir(parents=True, exist_ok=True)

        dataset.to_csv(self.config.ml_dataset_csv_path, index=False, encoding="utf-8-sig")
        predictions.to_csv(self.config.ml_predictions_csv_path, index=False, encoding="utf-8-sig")
        summary.to_csv(self.config.ml_summary_csv_path, index=False, encoding="utf-8-sig")

        resumo = pd.DataFrame(
            [
                {"campo": "dataset_rows", "valor": int(len(dataset))},
                {"campo": "predictions_rows", "valor": int(len(predictions))},
                {"campo": "summary_rows", "valor": int(len(summary))},
                {"campo": "train_ratio", "valor": float(train_ratio)},
                {"campo": "validation_ratio", "valor": float(validation_ratio)},
                {"campo": "epochs", "valor": int(epochs)},
                {"campo": "learning_rate", "valor": float(learning_rate)},
                {"campo": "l2", "valor": float(l2)},
                {"campo": "seed", "valor": int(seed)},
            ]
        )

        with pd.ExcelWriter(self.config.ml_excel_path, engine="openpyxl") as writer:
            predictions.to_excel(writer, index=False, sheet_name="predicoes")
            summary.to_excel(writer, index=False, sheet_name="resumo_modelos")
            resumo.to_excel(writer, index=False, sheet_name="resumo_execucao")

        test_predictions = predictions[predictions["split"] == "test"]
        self.logger.info("Dataset ML salvo em %s", self.config.ml_dataset_csv_path)
        self.logger.info("Predicoes ML salvas em %s", self.config.ml_predictions_csv_path)
        self.logger.info("Resumo ML salvo em %s", self.config.ml_summary_csv_path)
        self.logger.info("Excel ML salvo em %s", self.config.ml_excel_path)

        return MLSummary(
            dataset_rows=int(len(dataset)),
            predictions_rows=int(len(predictions)),
            summary_rows=int(len(summary)),
            first_test_concurso=int(test_predictions["concurso_previsto"].min()),
            last_test_concurso=int(test_predictions["concurso_previsto"].max()),
            dataset_csv_path=str(self.config.ml_dataset_csv_path),
            predictions_csv_path=str(self.config.ml_predictions_csv_path),
            summary_csv_path=str(self.config.ml_summary_csv_path),
            excel_path=str(self.config.ml_excel_path),
        )
