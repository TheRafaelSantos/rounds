from __future__ import annotations

import json
import logging
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import pandas as pd

from .config import AppConfig
from .normalize import DEZENAS
from .storage import sanitize_dataframe_for_tabular_output


TOP100_REPAIR_MODEL = "top100_near_miss_repair_learning_v1"


@dataclass(frozen=True)
class Top100RepairLearningSummary:
    status: str
    processed_this_run: int
    learned_contests: int
    best_hits_seen: int
    repair_weights_json_path: str
    state_json_path: str
    results_csv_path: str
    summary_csv_path: str
    message: str

    def to_console(self) -> str:
        return "\n".join(
            [
                "",
                "Resumo Lotofacil Analytics - Aprendizado de Reparo Top100",
                f"Status: {self.status}",
                f"Concursos aprendidos nesta execucao: {self.processed_this_run}",
                f"Concursos aprendidos no total: {self.learned_contests}",
                f"Melhor quase-acerto observado: {self.best_hits_seen}",
                f"Pesos de reparo: {self.repair_weights_json_path}",
                f"Estado retomavel: {self.state_json_path}",
                f"CSV resultados: {self.results_csv_path}",
                f"CSV resumo: {self.summary_csv_path}",
                f"Mensagem: {self.message}",
            ]
        )


def _parse_nums(text: str) -> tuple[int, ...]:
    nums = tuple(sorted(int(part) for part in str(text).split()))
    if len(nums) != 15 or len(set(nums)) != 15 or any(n < 1 or n > 25 for n in nums):
        raise ValueError(f"Jogo invalido: {text}")
    return nums


def _format_nums(nums: Sequence[int]) -> str:
    return " ".join(f"{int(n):02d}" for n in sorted(nums))


def _actual_from_row(row: pd.Series) -> tuple[int, ...]:
    return tuple(sorted(int(row[col]) for col in DEZENAS))


def _prediction_history_path(config: AppConfig, concurso: int) -> Path:
    return config.top100_prediction_history_dir / f"lotofacil_prediction_top100_concurso_{int(concurso):04d}.csv"


def archive_top100_prediction(config: AppConfig, *, concurso: int) -> Path:
    config.top100_prediction_history_dir.mkdir(parents=True, exist_ok=True)
    target = _prediction_history_path(config, int(concurso))
    if config.top100_prediction_csv_path.exists():
        shutil.copy2(config.top100_prediction_csv_path, target)
    return target


def load_top100_repair_payload(path: Path) -> Dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("model") != TOP100_REPAIR_MODEL:
        return None
    return payload


def _load_json(path: Path, default: Mapping[str, object]) -> Dict[str, object]:
    if not path.exists():
        return dict(default)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(default)
    return payload if isinstance(payload, dict) else dict(default)


def _normalize_counter(counter: Counter[int]) -> Dict[str, float]:
    if not counter:
        return {}
    max_value = max(float(value) for value in counter.values()) or 1.0
    return {f"{int(key):02d}": round(float(value) / max_value * 100.0, 6) for key, value in sorted(counter.items())}


def _merge_counter_from_json(value: object) -> Counter[int]:
    counter: Counter[int] = Counter()
    if not isinstance(value, dict):
        return counter
    for key, raw in value.items():
        try:
            counter[int(key)] = float(raw)
        except (TypeError, ValueError):
            continue
    return counter


def _json_counter(counter: Counter[int]) -> Dict[str, float]:
    return {str(int(key)): round(float(value), 6) for key, value in sorted(counter.items())}


def _score_hit_weight(hits: int) -> float:
    if hits >= 14:
        return 10.0
    if hits == 13:
        return 6.0
    if hits == 12:
        return 3.0
    if hits == 11:
        return 1.0
    return 0.0


def learn_from_prediction(
    prediction_df: pd.DataFrame,
    *,
    actual_numbers: Sequence[int],
    concurso: int,
    payload: Mapping[str, object] | None = None,
    min_hits: int = 11,
) -> tuple[Dict[str, object], pd.DataFrame]:
    actual = set(int(n) for n in actual_numbers)
    if len(actual) != 15:
        raise ValueError("Resultado real precisa ter 15 dezenas.")
    previous = payload or {}
    add_counter = _merge_counter_from_json(previous.get("add_counter"))
    remove_counter = _merge_counter_from_json(previous.get("remove_counter"))
    strategy_counter: Counter[str] = Counter(previous.get("strategy_counter", {}) if isinstance(previous.get("strategy_counter"), dict) else {})
    swap_counter: Counter[str] = Counter(previous.get("swap_counter", {}) if isinstance(previous.get("swap_counter"), dict) else {})

    rows: List[Dict[str, object]] = []
    if prediction_df.empty or "nums" not in prediction_df.columns:
        raise ValueError("Arquivo Top100 sem coluna nums.")
    for _, row in prediction_df.iterrows():
        nums = set(_parse_nums(str(row["nums"])))
        hits = len(nums & actual)
        missing = sorted(actual - nums)
        extras = sorted(nums - actual)
        weight = _score_hit_weight(hits)
        strategy = str(row.get("estrategia_origem_top100", row.get("source_model", "")) or "")
        rows.append(
            {
                "concurso": int(concurso),
                "rank_top100": int(row.get("rank_top100", 0) or 0),
                "hits": int(hits),
                "missing": _format_nums(missing) if missing else "",
                "extras": _format_nums(extras) if extras else "",
                "strategy": strategy,
                "learned": int(hits >= int(min_hits) and weight > 0.0),
            }
        )
        if hits < int(min_hits) or weight <= 0.0:
            continue
        for number in missing:
            add_counter[int(number)] += weight
        for number in extras:
            remove_counter[int(number)] += weight
        strategy_counter[strategy] += weight
        swap_counter[str(len(missing))] += weight

    learned_rows = [row for row in rows if int(row["learned"]) == 1]
    contests = int(previous.get("contests", 0) or 0) + 1
    learned_contests = set(str(item) for item in previous.get("learned_contests", []) if str(item).strip())
    learned_contests.add(str(int(concurso)))
    out = {
        "model": TOP100_REPAIR_MODEL,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "contests": contests,
        "learned_contests": sorted(learned_contests, key=lambda value: int(value)),
        "min_hits": int(min_hits),
        "best_hits_seen": max(max([int(row["hits"]) for row in rows] or [0]), int(previous.get("best_hits_seen", 0) or 0)),
        "add_counter": _json_counter(add_counter),
        "remove_counter": _json_counter(remove_counter),
        "add_scores": _normalize_counter(add_counter),
        "remove_scores": _normalize_counter(remove_counter),
        "strategy_counter": {str(key): round(float(value), 6) for key, value in sorted(strategy_counter.items())},
        "swap_counter": {str(key): round(float(value), 6) for key, value in sorted(swap_counter.items())},
        "last_concurso": int(concurso),
        "last_learned_rows": int(len(learned_rows)),
    }
    return out, pd.DataFrame(rows)


class Top100RepairLearningPipeline:
    def __init__(self, *, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def _prediction_files(self) -> Iterable[Path]:
        if not self.config.top100_prediction_history_dir.exists():
            return []
        return sorted(self.config.top100_prediction_history_dir.glob("lotofacil_prediction_top100_concurso_*.csv"))

    @staticmethod
    def _contest_from_path(path: Path) -> int | None:
        stem = path.stem
        try:
            return int(stem.rsplit("_", 1)[-1])
        except (TypeError, ValueError):
            return None

    def run(
        self,
        *,
        max_contests: int = 0,
        min_hits: int = 11,
        prediction_file: Path | None = None,
        actual_numbers: Sequence[int] | None = None,
        concurso: int | None = None,
    ) -> Top100RepairLearningSummary:
        self.config.processed_dir.mkdir(parents=True, exist_ok=True)
        state = _load_json(self.config.top100_repair_state_json_path, {"learned_contests": []})
        payload = load_top100_repair_payload(self.config.top100_repair_weights_json_path) or {}
        learned = set(str(value) for value in state.get("learned_contests", []) if str(value).strip())

        history = pd.read_csv(self.config.processed_csv_path) if self.config.processed_csv_path.exists() else pd.DataFrame()
        if not history.empty and "concurso" in history.columns:
            history["concurso"] = pd.to_numeric(history["concurso"], errors="coerce")

        result_frames: List[pd.DataFrame] = []
        processed = 0
        candidates: List[tuple[int, Path, Sequence[int]]] = []
        if prediction_file is not None:
            if concurso is None:
                raise ValueError("Informe concurso junto com prediction_file.")
            if actual_numbers is None:
                raise ValueError("Informe actual_numbers junto com prediction_file.")
            candidates.append((int(concurso), prediction_file, tuple(int(n) for n in actual_numbers)))
        else:
            for path in self._prediction_files():
                target = self._contest_from_path(path)
                if target is None or str(target) in learned:
                    continue
                if history.empty:
                    continue
                match = history[history["concurso"] == int(target)]
                if match.empty:
                    continue
                candidates.append((int(target), path, _actual_from_row(match.iloc[0])))

        for target, path, actual in candidates:
            if max_contests and processed >= int(max_contests):
                break
            prediction_df = pd.read_csv(path)
            payload, result_df = learn_from_prediction(
                prediction_df,
                actual_numbers=actual,
                concurso=target,
                payload=payload,
                min_hits=min_hits,
            )
            learned.add(str(int(target)))
            result_frames.append(result_df)
            processed += 1
            self.logger.info("Reparo Top100 aprendeu concurso %s a partir de %s", target, path)

        state.update(
            {
                "model": TOP100_REPAIR_MODEL,
                "status": "complete" if not candidates or processed == len(candidates) else "running",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "learned_contests": sorted(learned, key=lambda value: int(value)),
                "learned_contests_count": len(learned),
                "processed_this_run": processed,
                "pending_prediction_files": max(0, len(candidates) - processed),
                "best_hits_seen": int(payload.get("best_hits_seen", state.get("best_hits_seen", 0)) or 0),
                "min_hits": int(min_hits),
            }
        )
        self.config.top100_repair_state_json_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        if payload:
            self.config.top100_repair_weights_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        previous_results = pd.read_csv(self.config.top100_repair_results_csv_path) if self.config.top100_repair_results_csv_path.exists() else pd.DataFrame()
        current_results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
        all_results = pd.concat([previous_results, current_results], ignore_index=True) if not previous_results.empty or not current_results.empty else pd.DataFrame()
        if not all_results.empty:
            all_results = all_results.drop_duplicates(subset=["concurso", "rank_top100"], keep="last")
            sanitize_dataframe_for_tabular_output(all_results).to_csv(self.config.top100_repair_results_csv_path, index=False, encoding="utf-8-sig")

        summary_rows = [
            {"metrica": "learned_contests", "valor": len(learned)},
            {"metrica": "processed_this_run", "valor": processed},
            {"metrica": "best_hits_seen", "valor": int(state.get("best_hits_seen", 0) or 0)},
            {"metrica": "add_scores", "valor": json.dumps(payload.get("add_scores", {}), ensure_ascii=False) if payload else "{}"},
            {"metrica": "remove_scores", "valor": json.dumps(payload.get("remove_scores", {}), ensure_ascii=False) if payload else "{}"},
        ]
        pd.DataFrame(summary_rows).to_csv(self.config.top100_repair_summary_csv_path, index=False, encoding="utf-8-sig")
        return Top100RepairLearningSummary(
            status=str(state.get("status", "complete")),
            processed_this_run=processed,
            learned_contests=len(learned),
            best_hits_seen=int(state.get("best_hits_seen", 0) or 0),
            repair_weights_json_path=str(self.config.top100_repair_weights_json_path),
            state_json_path=str(self.config.top100_repair_state_json_path),
            results_csv_path=str(self.config.top100_repair_results_csv_path),
            summary_csv_path=str(self.config.top100_repair_summary_csv_path),
            message="aprende com jogos Top100 de concursos ja encerrados e cria memoria de trocas para concursos futuros.",
        )

    def status(self) -> Dict[str, object]:
        state = _load_json(self.config.top100_repair_state_json_path, {})
        payload = load_top100_repair_payload(self.config.top100_repair_weights_json_path) or {}
        return {
            "state": state,
            "add_scores": payload.get("add_scores", {}),
            "remove_scores": payload.get("remove_scores", {}),
            "strategy_counter": payload.get("strategy_counter", {}),
            "swap_counter": payload.get("swap_counter", {}),
        }

    def loop(self, *, sleep_seconds: int, max_contests: int, min_hits: int) -> None:
        while True:
            summary = self.run(max_contests=max_contests, min_hits=min_hits)
            print(summary.to_console(), flush=True)
            time.sleep(max(5, int(sleep_seconds)))
