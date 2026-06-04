from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import pandas as pd

from .backtest_lotofacil import compute_hits
from .exhaustive_optimizer import (
    DEFAULT_EXHAUSTIVE_WEIGHTS,
    build_exhaustive_candidates,
    format_exhaustive_weights,
    resolve_exhaustive_weights,
)
from .normalize import DEZENAS
from .predictor import select_final_games
from .storage import sanitize_dataframe_for_tabular_output


COMPONENT_SCORE_COLUMNS = {
    "estatistico": "score_estatistico",
    "historico": "score_historico",
    "atraso": "score_atraso",
    "combinatorio": "score_combinatorio",
    "localidade_numerologia": "score_localidade_numerologia",
    "climatico": "score_climatico",
    "temporal_profundo": "score_temporal_profundo",
    "cenarios": "score_cenarios",
    "contrarian": "score_contrarian",
    "transicao": "score_transicao",
}


@dataclass(frozen=True)
class CalibrationPilotSummary:
    target_concurso: int
    requested_attempts: int
    completed_attempts: int
    attempts_this_run: int
    best_hits: int
    best_attempt: int
    solved: bool
    elapsed_seconds: float
    candidate_base_seconds: float
    attempts_elapsed_seconds: float
    estimated_seconds_per_attempt: float
    candidates_csv_path: str
    results_csv_path: str
    summary_csv_path: str
    state_json_path: str
    excel_path: str

    def to_console(self) -> str:
        return "\n".join(
            [
                "",
                "Resumo Lotofacil Analytics - Piloto de Calibracao por Pesos",
                f"Concurso-alvo: {self.target_concurso}",
                f"Tentativas solicitadas: {self.requested_attempts}",
                f"Tentativas concluidas: {self.completed_attempts}",
                f"Tentativas nesta execucao: {self.attempts_this_run}",
                f"Melhor acerto: {self.best_hits}",
                f"Melhor tentativa: {self.best_attempt}",
                f"Encontrou 15 pontos: {'sim' if self.solved else 'nao'}",
                f"Tempo desta execucao: {self.elapsed_seconds:.2f}s",
                f"Tempo preparo/carregamento da base: {self.candidate_base_seconds:.2f}s",
                f"Tempo das tentativas nesta execucao: {self.attempts_elapsed_seconds:.2f}s",
                f"Media por tentativa nesta execucao: {self.estimated_seconds_per_attempt:.4f}s",
                f"CSV candidatos: {self.candidates_csv_path}",
                f"CSV resultados: {self.results_csv_path}",
                f"CSV resumo: {self.summary_csv_path}",
                f"Estado retomavel: {self.state_json_path}",
                f"Excel: {self.excel_path}",
                "Mensagem: Piloto local retomavel; se interromper, rode o mesmo comando para continuar.",
            ]
        )


def _nums_from_row(row: pd.Series) -> List[int]:
    return sorted(int(row[col]) for col in DEZENAS)


def _parse_nums(text: str) -> List[int]:
    return sorted(int(part) for part in str(text).split())


def _target_climate_from_features(climate_features: pd.DataFrame | None, concurso: int) -> Mapping[str, object] | None:
    if climate_features is None or climate_features.empty or "concurso" not in climate_features.columns:
        return None
    df = climate_features.copy()
    df["concurso"] = pd.to_numeric(df["concurso"], errors="coerce")
    rows = df[df["concurso"] == int(concurso)]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _component_weights_for_attempt(attempt: int, seed: int) -> Dict[str, float]:
    components = list(DEFAULT_EXHAUSTIVE_WEIGHTS)
    presets: List[Tuple[str, Dict[str, float]]] = [
        ("default", dict(DEFAULT_EXHAUSTIVE_WEIGHTS)),
        ("sem_localidade_numerologia", {**DEFAULT_EXHAUSTIVE_WEIGHTS, "localidade_numerologia": 0.0}),
        ("sem_clima", {**DEFAULT_EXHAUSTIVE_WEIGHTS, "climatico": 0.0}),
        ("sem_temporal_profundo", {**DEFAULT_EXHAUSTIVE_WEIGHTS, "temporal_profundo": 0.0}),
        ("sem_contextos_supersticiosos", {**DEFAULT_EXHAUSTIVE_WEIGHTS, "localidade_numerologia": 0.0, "climatico": 0.0, "temporal_profundo": 0.0}),
    ]
    if attempt <= len(presets):
        return resolve_exhaustive_weights(presets[attempt - 1][1])

    rng = random.Random(int(seed) + int(attempt) * 7919)
    values: Dict[str, float] = {}
    for component in components:
        base = float(DEFAULT_EXHAUSTIVE_WEIGHTS[component])
        if rng.random() < 0.12:
            values[component] = 0.0
        else:
            values[component] = rng.gammavariate(1.0 + base * 12.0, 1.0)
    return resolve_exhaustive_weights(values)


def _score_candidates_with_weights(candidates: pd.DataFrame, weights: Mapping[str, float]) -> pd.DataFrame:
    out = candidates.copy()
    score = pd.Series(0.0, index=out.index, dtype="float64")
    for component, weight in weights.items():
        if component == "nao_repeticao_exata":
            if "ja_saiu_exatamente_no_historico" in out.columns:
                component_score = out["ja_saiu_exatamente_no_historico"].map(lambda value: 92.0 if int(value) else 100.0)
            else:
                component_score = pd.Series(100.0, index=out.index, dtype="float64")
        else:
            column = COMPONENT_SCORE_COLUMNS.get(component)
            if column is None or column not in out.columns:
                component_score = pd.Series(50.0, index=out.index, dtype="float64")
            else:
                component_score = pd.to_numeric(out[column], errors="coerce").fillna(50.0)
        score = score + float(weight) * component_score
    out["score_final"] = score.round(6)
    out = out.sort_values(["score_final", "nums"], ascending=[False, True]).reset_index(drop=True)
    if "rank" in out.columns:
        out = out.drop(columns=["rank"])
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def _append_result(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(row)]).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
        encoding="utf-8-sig",
    )


def _build_candidate_base(
    *,
    train_df: pd.DataFrame,
    climate_features: pd.DataFrame | None,
    target_concurso: int,
    candidate_pool: int,
    exhaustive_limit: int | None,
    draw_hour: int,
    draw_minute: int,
    candidates_csv_path: Path,
) -> pd.DataFrame:
    if candidates_csv_path.exists():
        cached = pd.read_csv(candidates_csv_path)
        if not cached.empty and "nums" in cached.columns:
            return cached

    candidates, _summary = build_exhaustive_candidates(
        train_df,
        top_games=max(2, int(candidate_pool)),
        draw_hour=draw_hour,
        draw_minute=draw_minute,
        limit_combinations=exhaustive_limit,
        weights=DEFAULT_EXHAUSTIVE_WEIGHTS,
        climate_features=climate_features,
        target_climate=_target_climate_from_features(climate_features, target_concurso),
    )
    candidates = sanitize_dataframe_for_tabular_output(candidates)
    candidates_csv_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(candidates_csv_path, index=False, encoding="utf-8-sig")
    return candidates


def _summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame([{"metrica": "tentativas", "valor": 0}])
    best = results.sort_values(["melhor_acerto", "tentativa"], ascending=[False, True]).iloc[0]
    return pd.DataFrame(
        [
            {"metrica": "tentativas", "valor": int(len(results))},
            {"metrica": "melhor_acerto", "valor": int(best["melhor_acerto"])},
            {"metrica": "melhor_tentativa", "valor": int(best["tentativa"])},
            {"metrica": "tentativas_com_15", "valor": int((pd.to_numeric(results["melhor_acerto"], errors="coerce") >= 15).sum())},
            {"metrica": "media_melhor_acerto", "valor": round(float(pd.to_numeric(results["melhor_acerto"], errors="coerce").mean()), 6)},
            {"metrica": "media_tempo_tentativa_segundos", "valor": round(float(pd.to_numeric(results["elapsed_seconds"], errors="coerce").mean()), 6)},
            {"metrica": "melhor_jogo", "valor": str(best["melhor_jogo"])},
            {"metrica": "melhores_pesos", "valor": str(best["score_weights"])},
        ]
    )


def _save_outputs(results_path: Path, summary_path: Path, excel_path: Path) -> pd.DataFrame:
    results = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame()
    summary = _summarize_results(results)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        results.to_excel(writer, index=False, sheet_name="tentativas")
        summary.to_excel(writer, index=False, sheet_name="resumo")
    return summary


def run_calibration_pilot(
    concursos: pd.DataFrame,
    *,
    climate_features: pd.DataFrame | None,
    target_concurso: int,
    attempts: int,
    candidate_pool: int,
    exhaustive_limit: int | None,
    max_overlap: int,
    seed: int,
    draw_hour: int,
    draw_minute: int,
    candidates_csv_path: Path,
    results_csv_path: Path,
    summary_csv_path: Path,
    state_json_path: Path,
    excel_path: Path,
    reset: bool = False,
) -> CalibrationPilotSummary:
    if concursos.empty:
        raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")
    if int(attempts) <= 0:
        raise ValueError("--pilot-games deve ser maior que zero.")
    if int(candidate_pool) < 2:
        raise ValueError("--pilot-candidate-pool deve ser pelo menos 2.")

    if reset:
        for path in [candidates_csv_path, results_csv_path, summary_csv_path, state_json_path, excel_path]:
            if path.exists():
                path.unlink()

    df = concursos.copy().sort_values("concurso").reset_index(drop=True)
    matches = df.index[pd.to_numeric(df["concurso"], errors="coerce") == int(target_concurso)].tolist()
    if not matches:
        raise ValueError(f"Concurso {target_concurso} nao encontrado na base local.")
    target_idx = int(matches[0])
    if target_idx < 10:
        raise ValueError("Piloto exige pelo menos 10 concursos anteriores ao concurso-alvo.")

    train_df = df.iloc[:target_idx].copy()
    target_row = df.iloc[target_idx]
    actual_nums = _nums_from_row(target_row)

    started = time.perf_counter()
    candidate_base_started = time.perf_counter()
    candidates = _build_candidate_base(
        train_df=train_df,
        climate_features=climate_features,
        target_concurso=int(target_concurso),
        candidate_pool=int(candidate_pool),
        exhaustive_limit=exhaustive_limit,
        draw_hour=draw_hour,
        draw_minute=draw_minute,
        candidates_csv_path=candidates_csv_path,
    )
    candidate_base_elapsed = time.perf_counter() - candidate_base_started
    if candidates.empty:
        raise ValueError("Base de candidatos do piloto ficou vazia.")

    existing_results = pd.read_csv(results_csv_path) if results_csv_path.exists() else pd.DataFrame()
    completed_attempts = set()
    if not existing_results.empty and "tentativa" in existing_results.columns:
        completed_attempts = set(int(value) for value in pd.to_numeric(existing_results["tentativa"], errors="coerce").dropna())

    state = _load_json(state_json_path)
    state.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
    state["target_concurso"] = int(target_concurso)
    state["requested_attempts"] = int(attempts)
    state["candidate_pool"] = int(candidate_pool)
    state["exhaustive_limit"] = int(exhaustive_limit) if exhaustive_limit is not None else None
    state["candidates_csv_path"] = str(candidates_csv_path)
    state["results_csv_path"] = str(results_csv_path)

    attempts_this_run = 0
    attempts_elapsed = 0.0
    solved = bool(state.get("solved", False))
    for attempt in range(1, int(attempts) + 1):
        if attempt in completed_attempts:
            continue
        attempt_start = time.perf_counter()
        weights = _component_weights_for_attempt(attempt, seed)
        ranked = _score_candidates_with_weights(candidates, weights)
        final_games = select_final_games(ranked, max_overlap=max_overlap)
        game_1 = str(final_games.iloc[0]["nums"])
        game_2 = str(final_games.iloc[1]["nums"])
        hits_1 = compute_hits(_parse_nums(game_1), actual_nums)
        hits_2 = compute_hits(_parse_nums(game_2), actual_nums)
        best_hits = max(hits_1, hits_2)
        best_game = game_1 if hits_1 >= hits_2 else game_2
        elapsed = time.perf_counter() - attempt_start
        attempts_elapsed += elapsed
        row: Dict[str, object] = {
            "target_concurso": int(target_concurso),
            "tentativa": int(attempt),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "jogo_1": game_1,
            "acertos_jogo_1": int(hits_1),
            "jogo_2": game_2,
            "acertos_jogo_2": int(hits_2),
            "melhor_jogo": best_game,
            "melhor_acerto": int(best_hits),
            "encontrou_15": int(best_hits >= 15),
            "elapsed_seconds": round(float(elapsed), 6),
            "score_weights": format_exhaustive_weights(weights),
        }
        for component, value in weights.items():
            row[f"peso_{component}"] = round(float(value), 10)
        _append_result(results_csv_path, row)
        completed_attempts.add(attempt)
        attempts_this_run += 1

        if best_hits >= 15:
            solved = True
            state["solved"] = True
            state["solved_attempt"] = int(attempt)
            state["solved_at"] = datetime.now().isoformat(timespec="seconds")
            _write_json(state_json_path, state)
            break

        state["last_completed_attempt"] = int(attempt)
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _write_json(state_json_path, state)

    summary_df = _save_outputs(results_csv_path, summary_csv_path, excel_path)
    results = pd.read_csv(results_csv_path) if results_csv_path.exists() else pd.DataFrame()
    if results.empty:
        best_hits = 0
        best_attempt = 0
    else:
        best = results.sort_values(["melhor_acerto", "tentativa"], ascending=[False, True]).iloc[0]
        best_hits = int(best["melhor_acerto"])
        best_attempt = int(best["tentativa"])
    elapsed_total = time.perf_counter() - started
    avg_attempt = attempts_elapsed / attempts_this_run if attempts_this_run else 0.0
    state.update(
        {
            "completed_attempts": int(len(results)),
            "best_hits": int(best_hits),
            "best_attempt": int(best_attempt),
            "solved": bool(solved or best_hits >= 15),
            "last_candidate_base_seconds": round(float(candidate_base_elapsed), 6),
            "last_attempts_elapsed_seconds": round(float(attempts_elapsed), 6),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "summary_csv_path": str(summary_csv_path),
            "excel_path": str(excel_path),
        }
    )
    _write_json(state_json_path, state)
    return CalibrationPilotSummary(
        target_concurso=int(target_concurso),
        requested_attempts=int(attempts),
        completed_attempts=int(len(results)),
        attempts_this_run=int(attempts_this_run),
        best_hits=int(best_hits),
        best_attempt=int(best_attempt),
        solved=bool(solved or best_hits >= 15),
        elapsed_seconds=round(float(elapsed_total), 6),
        candidate_base_seconds=round(float(candidate_base_elapsed), 6),
        attempts_elapsed_seconds=round(float(attempts_elapsed), 6),
        estimated_seconds_per_attempt=round(float(avg_attempt), 6),
        candidates_csv_path=str(candidates_csv_path),
        results_csv_path=str(results_csv_path),
        summary_csv_path=str(summary_csv_path),
        state_json_path=str(state_json_path),
        excel_path=str(excel_path),
    )
