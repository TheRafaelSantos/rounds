from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping

import pandas as pd

from .config import AppConfig
from .supervised_calibration import load_supervised_calibration_status
from .supervised_calibration_pipeline import SupervisedCalibrationPipeline
from .top50_refinement_pipeline import Top50RefinementPipeline
from .top100_repair_learning import Top100RepairLearningPipeline
from .top100_walkforward_learning import Top100WalkForwardLearningPipeline


@dataclass(frozen=True)
class UnifiedLearningSummary:
    status: str
    cycle_started_at: str
    cycle_finished_at: str
    elapsed_seconds: float
    supervised_status: str
    top50_status: str
    top100_status: str
    repair_status: str
    state_json_path: str

    def to_console(self) -> str:
        return "\n".join(
            [
                "",
                "Resumo Lotofacil Analytics - Aprendizado Unificado",
                f"Status: {self.status}",
                f"Iniciado em: {self.cycle_started_at}",
                f"Finalizado em: {self.cycle_finished_at}",
                f"Tempo: {self.elapsed_seconds:.2f}s",
                f"Supervisionado: {self.supervised_status}",
                f"Top50: {self.top50_status}",
                f"Top100 walk-forward: {self.top100_status}",
                f"Reparo Top100: {self.repair_status}",
                f"Estado: {self.state_json_path}",
            ]
        )


def _now() -> str:
    return pd.Timestamp.now().isoformat(timespec="seconds")


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _env_int(name: str, default: int, *, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        value = int(str(raw).strip())
    except ValueError:
        return int(default)
    return max(int(minimum), int(value))


class _LearningFileLock:
    def __init__(self, path: Path, *, stale_seconds: int = 21600) -> None:
        self.path = path
        self.stale_seconds = int(stale_seconds)
        self.acquired = False

    def __enter__(self) -> "_LearningFileLock":
        now = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.mkdir(parents=False)
            self.acquired = True
            (self.path / "owner.json").write_text(
                json.dumps({"pid": os.getpid(), "started_at": _now()}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return self
        except FileExistsError:
            try:
                mtime = self.path.stat().st_mtime
            except OSError:
                mtime = now
            if now - mtime > self.stale_seconds:
                shutil.rmtree(self.path, ignore_errors=True)
                self.path.mkdir(parents=False)
                self.acquired = True
                (self.path / "owner.json").write_text(
                    json.dumps({"pid": os.getpid(), "started_at": _now(), "stale_lock_replaced": True}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return self
            self.acquired = False
            return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        if self.acquired:
            shutil.rmtree(self.path, ignore_errors=True)


class UnifiedLearningPipeline:
    def __init__(self, *, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def run_once(
        self,
        *,
        seed: int,
        draw_hour: int,
        draw_minute: int,
        supervised_max_contests: int | None = None,
        top50_max_contests: int | None = None,
        top100_max_contests: int | None = None,
        repair_max_contests: int | None = None,
        reset: bool = False,
    ) -> UnifiedLearningSummary:
        supervised_limit = int(supervised_max_contests) if supervised_max_contests is not None else _env_int("LOTOFACIL_LEARNING_SUPERVISED_MAX", 25, minimum=1)
        top50_limit = int(top50_max_contests) if top50_max_contests is not None else _env_int("LOTOFACIL_LEARNING_TOP50_MAX", 5, minimum=1)
        top100_limit = int(top100_max_contests) if top100_max_contests is not None else _env_int("LOTOFACIL_LEARNING_TOP100_MAX", 3, minimum=1)
        repair_limit = int(repair_max_contests) if repair_max_contests is not None else _env_int("LOTOFACIL_LEARNING_REPAIR_MAX", 10, minimum=1)
        top100_pool = _env_int("LOTOFACIL_LEARNING_TOP100_POOL", 500, minimum=100)
        top100_exhaustive_limit = _env_int("LOTOFACIL_LEARNING_TOP100_EXHAUSTIVE_LIMIT", 3000, minimum=100)
        top50_pool = _env_int("LOTOFACIL_LEARNING_TOP50_POOL", 2000, minimum=100)
        top50_exhaustive_limit = _env_int("LOTOFACIL_LEARNING_TOP50_EXHAUSTIVE_LIMIT", 50000, minimum=100)
        supervised_samples = _env_int("LOTOFACIL_LEARNING_SUPERVISED_SAMPLES", 800, minimum=10)

        started = time.perf_counter()
        started_at = _now()
        state = _load_json(self.config.unified_learning_state_json_path)
        state.update(
            {
                "status": "starting",
                "cycle_started_at": started_at,
                "seed": int(seed),
                "draw_hour": int(draw_hour),
                "draw_minute": int(draw_minute),
                "limits": {
                    "supervised_max_contests": int(supervised_limit),
                    "top50_max_contests": int(top50_limit),
                    "top100_max_contests": int(top100_limit),
                    "repair_max_contests": int(repair_limit),
                    "top100_pool": int(top100_pool),
                    "top100_exhaustive_limit": int(top100_exhaustive_limit),
                    "top50_pool": int(top50_pool),
                    "top50_exhaustive_limit": int(top50_exhaustive_limit),
                    "supervised_samples": int(supervised_samples),
                },
            }
        )
        _write_json(self.config.unified_learning_state_json_path, state)

        with _LearningFileLock(self.config.unified_learning_lock_dir) as lock:
            if not lock.acquired:
                state.update({"status": "already_running", "updated_at": _now()})
                _write_json(self.config.unified_learning_state_json_path, state)
                return UnifiedLearningSummary(
                    status="already_running",
                    cycle_started_at=started_at,
                    cycle_finished_at=_now(),
                    elapsed_seconds=round(float(time.perf_counter() - started), 6),
                    supervised_status=str(state.get("supervised_status", "")),
                    top50_status=str(state.get("top50_status", "")),
                    top100_status=str(state.get("top100_status", "")),
                    repair_status=str(state.get("repair_status", "")),
                    state_json_path=str(self.config.unified_learning_state_json_path),
                )

            state.update({"status": "running", "updated_at": _now()})
            _write_json(self.config.unified_learning_state_json_path, state)
            supervised_summary = SupervisedCalibrationPipeline(config=self.config, logger=self.logger).run(
                from_concurso=1,
                to_concurso=None,
                samples=supervised_samples,
                max_contests=supervised_limit,
                seed=seed,
                draw_hour=draw_hour,
                draw_minute=draw_minute,
                min_history=10,
                reset=reset,
            )
            state.update({"supervised_status": supervised_summary.status, "updated_at": _now()})
            _write_json(self.config.unified_learning_state_json_path, state)

            top50_summary = Top50RefinementPipeline(config=self.config, logger=self.logger).run(
                from_concurso=1,
                to_concurso=None,
                max_contests=top50_limit,
                min_history=300,
                top_pool=top50_pool,
                exhaustive_limit=top50_exhaustive_limit,
                seed=seed,
                draw_hour=draw_hour,
                draw_minute=draw_minute,
                reset=reset,
            )
            state.update({"top50_status": top50_summary.status, "updated_at": _now()})
            _write_json(self.config.unified_learning_state_json_path, state)

            top100_summary = Top100WalkForwardLearningPipeline(config=self.config, logger=self.logger).run(
                from_concurso=1,
                to_concurso=None,
                max_contests=top100_limit,
                min_history=10,
                top_count=100,
                top_pool=top100_pool,
                max_overlap=11,
                exhaustive_limit=top100_exhaustive_limit,
                seed=seed,
                draw_hour=draw_hour,
                draw_minute=draw_minute,
                reset=reset,
            )
            state.update({"top100_status": top100_summary.status, "updated_at": _now()})
            _write_json(self.config.unified_learning_state_json_path, state)

            repair_summary = Top100RepairLearningPipeline(config=self.config, logger=self.logger).run(
                max_contests=repair_limit,
                min_hits=11,
            )
            finished_at = _now()
            elapsed = round(float(time.perf_counter() - started), 6)
            state.update(
                {
                    "status": "complete",
                    "cycle_finished_at": finished_at,
                    "elapsed_seconds": elapsed,
                    "supervised_status": supervised_summary.status,
                    "top50_status": top50_summary.status,
                    "top100_status": top100_summary.status,
                    "repair_status": repair_summary.status,
                    "updated_at": finished_at,
                }
            )
            _write_json(self.config.unified_learning_state_json_path, state)

        return UnifiedLearningSummary(
            status=str(state.get("status", "complete")),
            cycle_started_at=started_at,
            cycle_finished_at=str(state.get("cycle_finished_at", _now())),
            elapsed_seconds=float(state.get("elapsed_seconds", round(float(time.perf_counter() - started), 6)) or 0.0),
            supervised_status=str(state.get("supervised_status", "")),
            top50_status=str(state.get("top50_status", "")),
            top100_status=str(state.get("top100_status", "")),
            repair_status=str(state.get("repair_status", "")),
            state_json_path=str(self.config.unified_learning_state_json_path),
        )

    def loop(self, *, sleep_seconds: int, seed: int, draw_hour: int, draw_minute: int) -> None:
        while True:
            try:
                summary = self.run_once(seed=seed, draw_hour=draw_hour, draw_minute=draw_minute, reset=False)
                print(summary.to_console(), flush=True)
            except Exception as exc:
                self.logger.exception("Falha no aprendizado unificado: %s", exc)
                state = _load_json(self.config.unified_learning_state_json_path)
                state.update({"status": "error", "error": str(exc), "updated_at": _now()})
                _write_json(self.config.unified_learning_state_json_path, state)
            time.sleep(max(5, int(sleep_seconds)))

    def status(self) -> Dict[str, object]:
        return {
            "unified": _load_json(self.config.unified_learning_state_json_path),
            "supervised": load_supervised_calibration_status(
                state_json_path=self.config.supervised_calibration_state_json_path,
                results_csv_path=self.config.supervised_calibration_results_csv_path,
                summary_csv_path=self.config.supervised_calibration_summary_csv_path,
                weights_csv_path=self.config.supervised_calibration_weights_csv_path,
                weights_json_path=self.config.supervised_calibration_weights_json_path,
            ),
            "top50": Top50RefinementPipeline(config=self.config, logger=self.logger).status(),
            "top100_walkforward": Top100WalkForwardLearningPipeline(config=self.config, logger=self.logger).status(),
            "top100_repair": Top100RepairLearningPipeline(config=self.config, logger=self.logger).status(),
            "lock_active": self.config.unified_learning_lock_dir.exists(),
        }
