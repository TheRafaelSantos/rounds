from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .caixa_client import CaixaLotofacilClient
from .config import AppConfig
from .normalize import normalize_contest
from .storage import (
    ensure_dirs,
    list_raw_files,
    load_processed_csv,
    load_raw_payload,
    load_state,
    records_to_dataframe,
    save_processed_outputs,
    save_raw_payload,
    save_state,
)
from .validators import validate_dataset


@dataclass(frozen=True)
class PipelineSummary:
    action: str
    latest_remote: Optional[int]
    first_local: Optional[int]
    last_local: Optional[int]
    total_local: int
    fetched: int
    csv_path: Path
    excel_path: Path
    state_path: Path
    message: str

    def to_console(self) -> str:
        lines = [
            "",
            "Resumo Lotofacil Analytics - Fase 1",
            f"Acao: {self.action}",
            f"Ultimo concurso remoto: {self.latest_remote if self.latest_remote is not None else '-'}",
            f"Primeiro concurso local: {self.first_local if self.first_local is not None else '-'}",
            f"Ultimo concurso local: {self.last_local if self.last_local is not None else '-'}",
            f"Total local: {self.total_local}",
            f"Concursos baixados nesta execucao: {self.fetched}",
            f"CSV: {self.csv_path}",
            f"Excel: {self.excel_path}",
            f"Estado: {self.state_path}",
            f"Mensagem: {self.message}",
        ]
        return "\n".join(lines)


class LotofacilPipeline:
    def __init__(self, *, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.client = CaixaLotofacilClient(
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            request_sleep_seconds=config.request_sleep_seconds,
            logger=logger,
        )

    def status(self) -> PipelineSummary:
        ensure_dirs([self.config.raw_dir, self.config.processed_dir, self.config.exports_dir, self.config.logs_dir])
        df = load_processed_csv(self.config.processed_csv_path)
        state = load_state(self.config.state_path)
        latest_remote = state.get("latest_remote")
        if latest_remote is None:
            latest_remote = int(self.client.fetch_latest()["numero"])

        return PipelineSummary(
            action="status",
            latest_remote=int(latest_remote) if latest_remote is not None else None,
            first_local=int(df["concurso"].min()) if not df.empty else None,
            last_local=int(df["concurso"].max()) if not df.empty else None,
            total_local=int(len(df)),
            fetched=0,
            csv_path=self.config.processed_csv_path,
            excel_path=self.config.excel_path,
            state_path=self.config.state_path,
            message="Estado local consultado.",
        )

    def update(
        self,
        *,
        force_full: bool,
        from_concurso: Optional[int] = None,
        to_concurso: Optional[int] = None,
    ) -> PipelineSummary:
        ensure_dirs([self.config.raw_dir, self.config.processed_dir, self.config.exports_dir, self.config.logs_dir])
        latest_payload = self.client.fetch_latest()
        latest_remote = int(latest_payload["numero"])
        self.logger.info("Ultimo concurso remoto informado pela CAIXA: %s", latest_remote)

        existing_df = load_processed_csv(self.config.processed_csv_path)
        last_local = int(existing_df["concurso"].max()) if not existing_df.empty else 0

        if from_concurso is not None or to_concurso is not None:
            start = int(from_concurso or 1)
            end = int(to_concurso or latest_remote)
        elif force_full or existing_df.empty:
            start = 1
            end = latest_remote
        else:
            start = last_local + 1
            end = latest_remote

        if start < 1:
            raise ValueError("--from-concurso deve ser maior ou igual a 1.")
        if end > latest_remote:
            raise ValueError(f"--to-concurso nao pode passar do ultimo remoto ({latest_remote}).")
        if end < start:
            self.logger.info("Base ja atualizada: ultimo local=%s, ultimo remoto=%s", last_local, latest_remote)
            df = self._build_dataframe_from_existing_or_raw(existing_df)
            self._save_outputs_and_state(df, latest_remote=latest_remote, fetched=0)
            return self._summary(
                action="full" if force_full else "update",
                latest_remote=latest_remote,
                df=df,
                fetched=0,
                message="Nenhum concurso novo para baixar.",
            )

        fetched_payloads: List[Dict[str, Any]] = []
        total_to_fetch = end - start + 1
        for idx, concurso in enumerate(range(start, end + 1), start=1):
            payload = latest_payload if concurso == latest_remote else self.client.fetch_contest(concurso)
            save_raw_payload(self.config.raw_dir, payload)
            fetched_payloads.append(payload)
            if idx == 1 or idx == total_to_fetch or idx % 50 == 0:
                self.logger.info("Baixados %s/%s concursos | atual=%s", idx, total_to_fetch, concurso)

        if force_full or existing_df.empty or start == 1:
            records = self._records_from_all_raw()
        else:
            records = existing_df.to_dict(orient="records")
            records.extend(self._normalize_payloads(fetched_payloads))

        rows = validate_dataset(records, require_contiguous=True)
        df = records_to_dataframe(rows)
        self._save_outputs_and_state(df, latest_remote=latest_remote, fetched=len(fetched_payloads))

        return self._summary(
            action="full" if force_full else "update",
            latest_remote=latest_remote,
            df=df,
            fetched=len(fetched_payloads),
            message="Base atualizada e validada.",
        )

    def _records_from_all_raw(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for path in list_raw_files(self.config.raw_dir):
            payload = load_raw_payload(path)
            records.append(normalize_contest(payload, raw_json_file=str(path.relative_to(self.config.base_dir))))
        return records

    def _normalize_payloads(self, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for payload in payloads:
            raw_path = self.config.raw_dir / f"concurso_{int(payload['numero']):06d}.json"
            records.append(normalize_contest(payload, raw_json_file=str(raw_path.relative_to(self.config.base_dir))))
        return records

    def _build_dataframe_from_existing_or_raw(self, existing_df: pd.DataFrame) -> pd.DataFrame:
        if not existing_df.empty:
            records = existing_df.to_dict(orient="records")
        else:
            records = self._records_from_all_raw()
        rows = validate_dataset(records, require_contiguous=True)
        return records_to_dataframe(rows)

    def _save_outputs_and_state(self, df: pd.DataFrame, *, latest_remote: int, fetched: int) -> None:
        save_processed_outputs(df, csv_path=self.config.processed_csv_path, excel_path=self.config.excel_path)
        state = {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "latest_remote": int(latest_remote),
            "first_local": int(df["concurso"].min()) if not df.empty else None,
            "last_local": int(df["concurso"].max()) if not df.empty else None,
            "total_local": int(len(df)),
            "fetched_last_run": int(fetched),
            "source": CaixaLotofacilClient.BASE_URL,
        }
        save_state(self.config.state_path, state)
        self.logger.info("CSV salvo em %s", self.config.processed_csv_path)
        self.logger.info("Excel salvo em %s", self.config.excel_path)

    def _summary(
        self,
        *,
        action: str,
        latest_remote: int,
        df: pd.DataFrame,
        fetched: int,
        message: str,
    ) -> PipelineSummary:
        return PipelineSummary(
            action=action,
            latest_remote=int(latest_remote),
            first_local=int(df["concurso"].min()) if not df.empty else None,
            last_local=int(df["concurso"].max()) if not df.empty else None,
            total_local=int(len(df)),
            fetched=int(fetched),
            csv_path=self.config.processed_csv_path,
            excel_path=self.config.excel_path,
            state_path=self.config.state_path,
            message=message,
        )
