from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from .normalize import column_order


ILLEGAL_EXCEL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]")


def sanitize_text_for_tabular_output(value: Any) -> Any:
    if isinstance(value, str):
        return ILLEGAL_EXCEL_CHARS_RE.sub("", value)
    return value


def sanitize_dataframe_for_tabular_output(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].map(sanitize_text_for_tabular_output)
    return out


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def raw_file_path(raw_dir: Path, concurso: int) -> Path:
    return raw_dir / f"concurso_{int(concurso):06d}.json"


def save_raw_payload(raw_dir: Path, payload: Dict[str, Any]) -> Path:
    concurso = int(payload["numero"])
    path = raw_file_path(raw_dir, concurso)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    return path


def load_raw_payload(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON bruto invalido: {path}")
    return payload


def list_raw_files(raw_dir: Path) -> List[Path]:
    if not raw_dir.exists():
        return []
    return sorted(raw_dir.glob("concurso_*.json"))


def load_processed_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def records_to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    cols = [c for c in column_order() if c in df.columns]
    extra = [c for c in df.columns if c not in cols]
    df = df[cols + extra]
    df = df.sort_values("concurso").reset_index(drop=True)
    return df


def save_processed_outputs(df: pd.DataFrame, *, csv_path: Path, excel_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    excel_path.parent.mkdir(parents=True, exist_ok=True)

    df = sanitize_dataframe_for_tabular_output(df)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    excel_df = df.copy()
    for col in ["data_sorteio", "data_proximo_concurso"]:
        if col in excel_df.columns:
            excel_df[col] = pd.to_datetime(excel_df[col], errors="coerce")

    resumo = pd.DataFrame(
        [
            {"campo": "gerado_em", "valor": datetime.now().isoformat(timespec="seconds")},
            {"campo": "qtd_concursos", "valor": int(len(df))},
            {"campo": "primeiro_concurso", "valor": int(df["concurso"].min()) if len(df) else None},
            {"campo": "ultimo_concurso", "valor": int(df["concurso"].max()) if len(df) else None},
        ]
    )

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        excel_df.to_excel(writer, index=False, sheet_name="concursos")
        resumo.to_excel(writer, index=False, sheet_name="resumo")


def save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)


def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        state = json.load(f)
    return state if isinstance(state, dict) else {}
