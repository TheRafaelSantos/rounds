from __future__ import annotations

import logging

import pandas as pd

from .config import AppConfig
from .storage import load_processed_csv, sanitize_dataframe_for_tabular_output
from .temporal_deep import TemporalDeepSummary, build_temporal_deep_rows, summarize_temporal_deep


def _existing_concursos(df: pd.DataFrame) -> set[int]:
    if df.empty or "concurso" not in df.columns:
        return set()
    values = pd.to_numeric(df["concurso"], errors="coerce").dropna()
    return {int(value) for value in values}


class TemporalDeepPipeline:
    def __init__(self, *, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def run(self, *, force: bool = False) -> TemporalDeepSummary:
        concursos = load_processed_csv(self.config.processed_csv_path)
        if concursos.empty:
            raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")
        existing = pd.DataFrame()
        if not force and self.config.temporal_deep_csv_path.exists():
            existing = pd.read_csv(self.config.temporal_deep_csv_path, encoding="utf-8-sig")

        all_concursos = _existing_concursos(concursos)
        done = set() if force else _existing_concursos(existing)
        missing = all_concursos - done
        new_rows = build_temporal_deep_rows(concursos, target_concursos=missing) if missing else pd.DataFrame()
        if existing.empty:
            rows = new_rows.copy()
        elif new_rows.empty:
            rows = existing.copy()
        else:
            rows = pd.concat([existing, new_rows], ignore_index=True)
            rows["concurso"] = pd.to_numeric(rows["concurso"], errors="coerce")
            rows["dezena"] = pd.to_numeric(rows["dezena"], errors="coerce")
            rows = rows.dropna(subset=["concurso", "dezena"]).copy()
            rows["concurso"] = rows["concurso"].astype(int)
            rows["dezena"] = rows["dezena"].astype(int)
            rows = rows.sort_values(["concurso", "dezena"]).drop_duplicates(["concurso", "dezena"], keep="last")

        rows = sanitize_dataframe_for_tabular_output(rows)
        summary = sanitize_dataframe_for_tabular_output(summarize_temporal_deep(rows))
        self.config.temporal_deep_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.temporal_deep_excel_path.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(self.config.temporal_deep_csv_path, index=False, encoding="utf-8-sig")
        summary.to_csv(self.config.temporal_deep_summary_csv_path, index=False, encoding="utf-8-sig")
        with pd.ExcelWriter(self.config.temporal_deep_excel_path, engine="openpyxl") as writer:
            rows.to_excel(writer, index=False, sheet_name="temporal_profundo")
            summary.to_excel(writer, index=False, sheet_name="resumo")

        self.logger.info("Temporal profundo salvo em %s", self.config.temporal_deep_csv_path)
        return TemporalDeepSummary(
            rows=int(len(rows)),
            contests_processed=int(len(missing)),
            contests_total=int(len(all_concursos)),
            csv_path=str(self.config.temporal_deep_csv_path),
            excel_path=str(self.config.temporal_deep_excel_path),
        )
