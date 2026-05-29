from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    exports_dir: Path
    logs_dir: Path
    timeout_seconds: float = 30.0
    max_retries: int = 3
    request_sleep_seconds: float = 0.05

    @classmethod
    def from_base_dir(
        cls,
        base_dir: Path,
        *,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        request_sleep_seconds: float = 0.05,
    ) -> "AppConfig":
        data_dir = base_dir / "data"
        return cls(
            base_dir=base_dir,
            data_dir=data_dir,
            raw_dir=data_dir / "raw" / "lotofacil",
            processed_dir=data_dir / "processed",
            exports_dir=data_dir / "exports",
            logs_dir=base_dir / "logs",
            timeout_seconds=timeout_seconds,
            max_retries=max(1, int(max_retries)),
            request_sleep_seconds=max(0.0, float(request_sleep_seconds)),
        )

    @property
    def processed_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_concursos.csv"

    @property
    def excel_path(self) -> Path:
        return self.exports_dir / "lotofacil_historico.xlsx"

    @property
    def state_path(self) -> Path:
        return self.processed_dir / "lotofacil_state.json"

    @property
    def features_base_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_features_base.csv"

    @property
    def features_base_excel_path(self) -> Path:
        return self.exports_dir / "lotofacil_features_base.xlsx"
