from __future__ import annotations

import logging

import pandas as pd

from .climate_features import ClimateSummary, build_climate_features
from .config import AppConfig
from .storage import load_processed_csv, sanitize_dataframe_for_tabular_output


class ClimatePipeline:
    def __init__(self, *, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def run(
        self,
        *,
        draw_hour: int = 20,
        draw_minute: int = 0,
        max_locations: int = 0,
        force: bool = False,
    ) -> ClimateSummary:
        concursos = load_processed_csv(self.config.processed_csv_path)
        if concursos.empty:
            raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")

        climate, summary = build_climate_features(
            concursos,
            cache_dir=self.config.climate_cache_dir,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
            timeout_seconds=self.config.timeout_seconds,
            retries=self.config.max_retries,
            sleep_seconds=self.config.request_sleep_seconds,
            max_locations=max_locations,
            force=force,
            logger=self.logger,
        )
        climate = sanitize_dataframe_for_tabular_output(climate)
        summary = sanitize_dataframe_for_tabular_output(summary)

        self.config.climate_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.climate_excel_path.parent.mkdir(parents=True, exist_ok=True)
        climate.to_csv(self.config.climate_csv_path, index=False, encoding="utf-8-sig")
        summary.to_csv(self.config.climate_summary_csv_path, index=False, encoding="utf-8-sig")

        with pd.ExcelWriter(self.config.climate_excel_path, engine="openpyxl") as writer:
            climate.to_excel(writer, index=False, sheet_name="clima")
            summary.to_excel(writer, index=False, sheet_name="resumo")

        locations_processed = int(len(summary))
        failed_locations = int((summary.get("status", pd.Series(dtype=object)).astype(str) == "erro").sum()) if not summary.empty else 0
        geocoded_locations = locations_processed - failed_locations
        locations_total = int(summary["localidades_total"].iloc[0]) if "localidades_total" in summary.columns and not summary.empty else locations_processed

        self.logger.info("Camada climatica salva em %s", self.config.climate_csv_path)
        self.logger.info("Resumo climatico salvo em %s", self.config.climate_summary_csv_path)
        return ClimateSummary(
            rows=int(len(climate)),
            locations_processed=locations_processed,
            locations_total=locations_total,
            geocoded_locations=geocoded_locations,
            failed_locations=failed_locations,
            csv_path=str(self.config.climate_csv_path),
            excel_path=str(self.config.climate_excel_path),
        )
