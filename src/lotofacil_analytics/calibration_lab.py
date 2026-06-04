from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

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


WEIGHT_COMPONENTS = tuple(DEFAULT_EXHAUSTIVE_WEIGHTS.keys())


@dataclass(frozen=True)
class CalibrationLabSummary:
    status: str
    current_concurso: int | None
    attempts_this_run: int
    total_attempts: int
    solved_contests: int
    best_hits_current: int
    best_game_current: str
    elapsed_seconds: float
    attempts_csv_path: str
    winners_csv_path: str
    state_json_path: str
    average_weights_csv_path: str
    engine_weights_json_path: str

    def to_console(self) -> str:
        return "\n".join(
            [
                "",
                "Resumo Lotofacil Analytics - Calibracao 24/7",
                f"Status: {self.status}",
                f"Concurso em analise: {self.current_concurso if self.current_concurso is not None else 'nenhum'}",
                f"Tentativas nesta execucao: {self.attempts_this_run}",
                f"Tentativas totais registradas: {self.total_attempts}",
                f"Concursos resolvidos com 15 pontos: {self.solved_contests}",
                f"Melhor acerto do concurso atual: {self.best_hits_current}",
                f"Melhor jogo do concurso atual: {self.best_game_current or '-'}",
                f"Tempo desta execucao: {self.elapsed_seconds:.2f}s",
                f"CSV tentativas: {self.attempts_csv_path}",
                f"CSV concursos resolvidos: {self.winners_csv_path}",
                f"Estado retomavel: {self.state_json_path}",
                f"Media de pesos vencedores: {self.average_weights_csv_path}",
                f"Pesos aplicados no motor principal: {self.engine_weights_json_path}",
                "Mensagem: processo retomavel; se interromper, rode o mesmo comando para continuar.",
            ]
        )


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _nums_from_row(row: pd.Series) -> List[int]:
    return sorted(int(row[col]) for col in DEZENAS)


def _parse_nums(text: str) -> List[int]:
    return sorted(int(part) for part in str(text).split())


def _format_nums(nums: Sequence[int]) -> str:
    return " ".join(f"{int(n):02d}" for n in sorted(nums))


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


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _append_csv(path: Path, row: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(row)]).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
        encoding="utf-8-sig",
    )


def _target_climate_from_features(climate_features: pd.DataFrame | None, concurso: int) -> Mapping[str, object] | None:
    if climate_features is None or climate_features.empty or "concurso" not in climate_features.columns:
        return None
    df = climate_features.copy()
    df["concurso"] = pd.to_numeric(df["concurso"], errors="coerce")
    rows = df[df["concurso"] == int(concurso)]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _delete_if_exists(paths: Sequence[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def _solved_contests(winners: pd.DataFrame) -> set[int]:
    if winners.empty or "target_concurso" not in winners.columns:
        return set()
    values = pd.to_numeric(winners["target_concurso"], errors="coerce").dropna()
    return set(int(value) for value in values)


def _next_target(contests: Sequence[int], solved: set[int], preferred: int | None) -> int | None:
    ordered = [int(value) for value in contests if int(value) not in solved]
    if not ordered:
        return None
    if preferred is not None and int(preferred) in ordered:
        return int(preferred)
    return int(ordered[0])


def _last_attempt_for_target(attempts: pd.DataFrame, target_concurso: int) -> int:
    if attempts.empty or "target_concurso" not in attempts.columns or "tentativa" not in attempts.columns:
        return 0
    target_rows = attempts[pd.to_numeric(attempts["target_concurso"], errors="coerce") == int(target_concurso)]
    if target_rows.empty:
        return 0
    values = pd.to_numeric(target_rows["tentativa"], errors="coerce").dropna()
    return int(values.max()) if len(values) else 0


def _best_for_target(attempts: pd.DataFrame, target_concurso: int) -> Dict[str, object]:
    if attempts.empty or "target_concurso" not in attempts.columns:
        return {"hits": 0, "game": "", "attempt": 0, "weights": {}}
    rows = attempts[pd.to_numeric(attempts["target_concurso"], errors="coerce") == int(target_concurso)].copy()
    if rows.empty or "melhor_acerto" not in rows.columns:
        return {"hits": 0, "game": "", "attempt": 0, "weights": {}}
    rows["melhor_acerto_numeric"] = pd.to_numeric(rows["melhor_acerto"], errors="coerce").fillna(0)
    rows["tentativa_numeric"] = pd.to_numeric(rows["tentativa"], errors="coerce").fillna(0)
    best = rows.sort_values(["melhor_acerto_numeric", "tentativa_numeric"], ascending=[False, True]).iloc[0]
    weights: Dict[str, float] = {}
    for component in WEIGHT_COMPONENTS:
        column = f"peso_{component}"
        if column in best and pd.notna(best[column]):
            weights[component] = float(best[column])
    return {
        "hits": int(best["melhor_acerto_numeric"]),
        "game": str(best.get("melhor_jogo", "")),
        "attempt": int(best["tentativa_numeric"]),
        "weights": weights,
    }


def _average_winner_weights(winners: pd.DataFrame) -> Dict[str, float] | None:
    if winners.empty:
        return None
    means: Dict[str, float] = {}
    for component in WEIGHT_COMPONENTS:
        column = f"peso_{component}"
        if column not in winners.columns:
            continue
        values = pd.to_numeric(winners[column], errors="coerce").dropna()
        if len(values):
            means[component] = float(values.mean())
    if not means:
        return None
    return resolve_exhaustive_weights(means)


def _write_average_outputs(
    *,
    winners: pd.DataFrame,
    average_weights_csv_path: Path,
    engine_weights_json_path: Path,
) -> Dict[str, float] | None:
    average = _average_winner_weights(winners)
    average_weights_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if average is None:
        pd.DataFrame(columns=["componente", "peso", "peso_percentual"]).to_csv(
            average_weights_csv_path,
            index=False,
            encoding="utf-8-sig",
        )
        return None

    rows = [
        {
            "componente": component,
            "peso": round(float(average[component]), 10),
            "peso_percentual": round(float(average[component]) * 100.0, 6),
        }
        for component in WEIGHT_COMPONENTS
    ]
    pd.DataFrame(rows).to_csv(average_weights_csv_path, index=False, encoding="utf-8-sig")
    contests = []
    if "target_concurso" in winners.columns:
        contests = [
            int(value)
            for value in pd.to_numeric(winners["target_concurso"], errors="coerce").dropna().tolist()
        ]
    payload = {
        "source": "calibration_lab_winners_average_v1",
        "updated_at": _now(),
        "solved_contests": contests,
        "solved_count": int(len(contests)),
        "weights": {component: round(float(average[component]), 10) for component in WEIGHT_COMPONENTS},
        "score_weights": format_exhaustive_weights(average),
        "note": "Media dos pesos das tentativas que acertaram 15 dezenas em um dos dois jogos.",
    }
    _write_json(engine_weights_json_path, payload)
    return average


def _write_summary(
    *,
    summary_csv_path: Path,
    state: Mapping[str, object],
    attempts: pd.DataFrame,
    winners: pd.DataFrame,
) -> None:
    rows = [
        {"metrica": "status", "valor": state.get("status", "")},
        {"metrica": "current_concurso", "valor": state.get("current_concurso", "")},
        {"metrica": "current_attempt", "valor": state.get("current_attempt", "")},
        {"metrica": "total_attempts", "valor": int(len(attempts))},
        {"metrica": "solved_contests", "valor": int(len(_solved_contests(winners)))},
        {"metrica": "best_hits_current", "valor": state.get("best_hits_current", 0)},
        {"metrica": "best_game_current", "valor": state.get("best_game_current", "")},
        {"metrica": "updated_at", "valor": state.get("updated_at", "")},
    ]
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(summary_csv_path, index=False, encoding="utf-8-sig")


def _write_excel_snapshot(
    *,
    excel_path: Path,
    attempts: pd.DataFrame,
    winners: pd.DataFrame,
    average_weights_csv_path: Path,
    summary_csv_path: Path,
) -> None:
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    summary = _read_csv(summary_csv_path)
    average = _read_csv(average_weights_csv_path)
    attempts_tail = attempts.tail(500).copy() if not attempts.empty else attempts
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="resumo")
        average.to_excel(writer, index=False, sheet_name="pesos_medios")
        winners.to_excel(writer, index=False, sheet_name="vencedores")
        attempts_tail.to_excel(writer, index=False, sheet_name="ultimas_tentativas")


def _base_weight_presets() -> List[Dict[str, float]]:
    return [
        dict(DEFAULT_EXHAUSTIVE_WEIGHTS),
        {**DEFAULT_EXHAUSTIVE_WEIGHTS, "localidade_numerologia": 0.0},
        {**DEFAULT_EXHAUSTIVE_WEIGHTS, "climatico": 0.0},
        {**DEFAULT_EXHAUSTIVE_WEIGHTS, "temporal_profundo": 0.0},
        {
            **DEFAULT_EXHAUSTIVE_WEIGHTS,
            "localidade_numerologia": 0.0,
            "climatico": 0.0,
            "temporal_profundo": 0.0,
        },
        {
            **DEFAULT_EXHAUSTIVE_WEIGHTS,
            "combinatorio": 0.22,
            "transicao": 0.20,
            "historico": 0.16,
            "localidade_numerologia": 0.04,
        },
        {
            **DEFAULT_EXHAUSTIVE_WEIGHTS,
            "contrarian": 0.22,
            "cenarios": 0.18,
            "nao_repeticao_exata": 0.10,
            "localidade_numerologia": 0.04,
        },
        {
            **DEFAULT_EXHAUSTIVE_WEIGHTS,
            "temporal_profundo": 0.18,
            "historico": 0.18,
            "atraso": 0.12,
            "localidade_numerologia": 0.05,
        },
    ]


def _weights_for_attempt(
    *,
    target_concurso: int,
    attempt: int,
    seed: int,
    average_winner_weights: Mapping[str, float] | None,
    best_current_weights: Mapping[str, float] | None,
) -> Dict[str, float]:
    presets = _base_weight_presets()
    if int(attempt) <= len(presets):
        return resolve_exhaustive_weights(presets[int(attempt) - 1])

    rng = random.Random(int(seed) + int(target_concurso) * 1000003 + int(attempt) * 7919)
    anchor: Mapping[str, float] | None = None
    if average_winner_weights and attempt % 3 == 0:
        anchor = average_winner_weights
    elif best_current_weights and attempt % 5 == 0:
        anchor = best_current_weights

    values: Dict[str, float] = {}
    if anchor:
        for component in WEIGHT_COMPONENTS:
            base = max(0.0001, float(anchor.get(component, DEFAULT_EXHAUSTIVE_WEIGHTS[component])))
            jitter = rng.lognormvariate(0.0, 0.65)
            values[component] = base * jitter
            if rng.random() < 0.05:
                values[component] = 0.0
    else:
        for component in WEIGHT_COMPONENTS:
            base = float(DEFAULT_EXHAUSTIVE_WEIGHTS[component])
            if rng.random() < 0.10:
                values[component] = 0.0
            else:
                values[component] = rng.gammavariate(0.7 + base * 14.0, 1.0)
    return resolve_exhaustive_weights(values)


def _evaluate_attempt(
    *,
    train_df: pd.DataFrame,
    target_concurso: int,
    actual_nums: Sequence[int],
    climate_features: pd.DataFrame | None,
    weights: Mapping[str, float],
    top_games: int,
    exhaustive_limit: int | None,
    max_overlap: int,
    draw_hour: int,
    draw_minute: int,
) -> Dict[str, object]:
    candidates, _summary = build_exhaustive_candidates(
        train_df,
        top_games=max(2, int(top_games)),
        draw_hour=draw_hour,
        draw_minute=draw_minute,
        limit_combinations=exhaustive_limit,
        weights=weights,
        climate_features=climate_features,
        target_climate=_target_climate_from_features(climate_features, int(target_concurso)),
    )
    candidates = sanitize_dataframe_for_tabular_output(candidates)
    if candidates.empty:
        raise ValueError("Motor exaustivo nao gerou candidatos para a tentativa.")

    final_games = select_final_games(candidates, max_overlap=max_overlap)
    game_1 = str(final_games.iloc[0]["nums"])
    game_2 = str(final_games.iloc[1]["nums"])
    hits_1 = compute_hits(_parse_nums(game_1), actual_nums)
    hits_2 = compute_hits(_parse_nums(game_2), actual_nums)
    best_hits = max(hits_1, hits_2)
    best_game = game_1 if hits_1 >= hits_2 else game_2
    return {
        "jogo_1": game_1,
        "acertos_jogo_1": int(hits_1),
        "jogo_2": game_2,
        "acertos_jogo_2": int(hits_2),
        "melhor_jogo": best_game,
        "melhor_acerto": int(best_hits),
        "candidates_evaluated": int(candidates.iloc[0].get("total_combinacoes_avaliadas", 0)),
        "score_jogo_1": float(final_games.iloc[0].get("score_final", 0.0)),
        "score_jogo_2": float(final_games.iloc[1].get("score_final", 0.0)),
    }


def load_calibration_lab_status(
    *,
    state_json_path: Path,
    attempts_csv_path: Path,
    winners_csv_path: Path,
    average_weights_csv_path: Path,
    engine_weights_json_path: Path,
    recent_rows: int = 12,
) -> Dict[str, object]:
    state = _load_json(state_json_path)
    attempts = _read_csv(attempts_csv_path)
    winners = _read_csv(winners_csv_path)
    average = _read_csv(average_weights_csv_path)
    engine_payload = _load_json(engine_weights_json_path)
    return {
        "state": state,
        "recent_attempts": attempts.tail(int(recent_rows)).astype(object).where(pd.notna(attempts.tail(int(recent_rows))), None).to_dict(orient="records") if not attempts.empty else [],
        "winners": winners.tail(int(recent_rows)).astype(object).where(pd.notna(winners.tail(int(recent_rows))), None).to_dict(orient="records") if not winners.empty else [],
        "average_weights": average.astype(object).where(pd.notna(average), None).to_dict(orient="records") if not average.empty else [],
        "engine_weights": engine_payload.get("weights", {}) if isinstance(engine_payload, dict) else {},
        "paths": {
            "state_json_path": str(state_json_path),
            "attempts_csv_path": str(attempts_csv_path),
            "winners_csv_path": str(winners_csv_path),
            "average_weights_csv_path": str(average_weights_csv_path),
            "engine_weights_json_path": str(engine_weights_json_path),
        },
    }


def run_calibration_lab(
    concursos: pd.DataFrame,
    *,
    climate_features: pd.DataFrame | None,
    from_concurso: int,
    to_concurso: int | None,
    max_attempts: int,
    top_games: int,
    exhaustive_limit: int | None,
    max_overlap: int,
    seed: int,
    draw_hour: int,
    draw_minute: int,
    min_history: int,
    max_runtime_seconds: int,
    reset: bool,
    state_json_path: Path,
    attempts_csv_path: Path,
    winners_csv_path: Path,
    summary_csv_path: Path,
    average_weights_csv_path: Path,
    excel_path: Path,
    engine_weights_json_path: Path,
) -> CalibrationLabSummary:
    if concursos.empty:
        raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")
    if int(top_games) < 2:
        raise ValueError("--lab-top-games deve ser pelo menos 2.")
    if int(max_overlap) < 0 or int(max_overlap) > 15:
        raise ValueError("--max-overlap-final deve estar entre 0 e 15.")

    if reset:
        _delete_if_exists(
            [
                state_json_path,
                attempts_csv_path,
                winners_csv_path,
                summary_csv_path,
                average_weights_csv_path,
                excel_path,
            ]
        )

    started_perf = time.perf_counter()
    started_at = _now()
    df = concursos.copy().sort_values("concurso").reset_index(drop=True)
    df["concurso"] = pd.to_numeric(df["concurso"], errors="coerce").astype("Int64")
    contests = [
        int(value)
        for value in df["concurso"].dropna().astype(int).tolist()
        if int(value) >= int(from_concurso) and (to_concurso is None or int(value) <= int(to_concurso))
    ]
    if not contests:
        raise ValueError("Nenhum concurso encontrado no intervalo solicitado para calibracao.")

    attempts_this_run = 0
    status = "running"
    state = _load_json(state_json_path)
    state.setdefault("created_at", started_at)
    state["started_last_run_at"] = started_at
    state["from_concurso"] = int(from_concurso)
    state["to_concurso"] = int(to_concurso) if to_concurso is not None else None
    state["top_games"] = int(top_games)
    state["exhaustive_limit"] = int(exhaustive_limit) if exhaustive_limit is not None else None
    state["max_overlap"] = int(max_overlap)
    state["draw_hour"] = int(draw_hour)
    state["draw_minute"] = int(draw_minute)
    state["status"] = status

    while True:
        attempts = _read_csv(attempts_csv_path)
        winners = _read_csv(winners_csv_path)
        average_winner_weights = _write_average_outputs(
            winners=winners,
            average_weights_csv_path=average_weights_csv_path,
            engine_weights_json_path=engine_weights_json_path,
        )
        solved = _solved_contests(winners)
        preferred = state.get("current_concurso")
        preferred_int = int(preferred) if preferred not in (None, "") else None
        target_concurso = _next_target(contests, solved, preferred_int)
        if target_concurso is None:
            status = "complete"
            state.update(
                {
                    "status": status,
                    "current_concurso": None,
                    "updated_at": _now(),
                    "message": "Todos os concursos do intervalo foram resolvidos com 15 pontos.",
                }
            )
            _write_json(state_json_path, state)
            _write_summary(summary_csv_path=summary_csv_path, state=state, attempts=attempts, winners=winners)
            break

        matches = df.index[df["concurso"].astype(int) == int(target_concurso)].tolist()
        if not matches:
            raise ValueError(f"Concurso {target_concurso} nao encontrado na base local.")
        target_idx = int(matches[0])
        if target_idx < int(min_history):
            solved.add(int(target_concurso))
            state["current_concurso"] = _next_target(contests, solved, None)
            state["updated_at"] = _now()
            _write_json(state_json_path, state)
            continue

        train_df = df.iloc[:target_idx].copy()
        target_row = df.iloc[target_idx]
        actual_nums = _nums_from_row(target_row)
        next_attempt = _last_attempt_for_target(attempts, target_concurso) + 1
        best_current = _best_for_target(attempts, target_concurso)
        weights = _weights_for_attempt(
            target_concurso=target_concurso,
            attempt=next_attempt,
            seed=seed,
            average_winner_weights=average_winner_weights,
            best_current_weights=best_current.get("weights", {}),
        )

        state.update(
            {
                "status": status,
                "current_concurso": int(target_concurso),
                "current_attempt": int(next_attempt),
                "best_hits_current": int(best_current.get("hits", 0)),
                "best_game_current": str(best_current.get("game", "")),
                "best_attempt_current": int(best_current.get("attempt", 0)),
                "solved_contests": sorted(solved),
                "solved_count": int(len(solved)),
                "average_weights": average_winner_weights or {},
                "last_attempt_started_at": _now(),
                "updated_at": _now(),
            }
        )
        state.pop("current_actual_numbers", None)
        _write_json(state_json_path, state)

        attempt_started = time.perf_counter()
        evaluation = _evaluate_attempt(
            train_df=train_df,
            target_concurso=target_concurso,
            actual_nums=actual_nums,
            climate_features=climate_features,
            weights=weights,
            top_games=top_games,
            exhaustive_limit=exhaustive_limit,
            max_overlap=max_overlap,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
        )
        elapsed_attempt = time.perf_counter() - attempt_started
        attempts_this_run += 1

        row: Dict[str, object] = {
            "target_concurso": int(target_concurso),
            "data_sorteio": str(target_row.get("data_sorteio", "")),
            "tentativa": int(next_attempt),
            "generated_at": _now(),
            "actual_numbers": _format_nums(actual_nums),
            "jogo_1": evaluation["jogo_1"],
            "acertos_jogo_1": int(evaluation["acertos_jogo_1"]),
            "jogo_2": evaluation["jogo_2"],
            "acertos_jogo_2": int(evaluation["acertos_jogo_2"]),
            "melhor_jogo": evaluation["melhor_jogo"],
            "melhor_acerto": int(evaluation["melhor_acerto"]),
            "encontrou_15": int(evaluation["melhor_acerto"] >= 15),
            "elapsed_seconds": round(float(elapsed_attempt), 6),
            "candidates_evaluated": int(evaluation["candidates_evaluated"]),
            "top_games": int(top_games),
            "exhaustive_limit": int(exhaustive_limit) if exhaustive_limit is not None else "",
            "max_overlap": int(max_overlap),
            "score_jogo_1": float(evaluation["score_jogo_1"]),
            "score_jogo_2": float(evaluation["score_jogo_2"]),
            "score_weights": format_exhaustive_weights(weights),
        }
        for component, value in weights.items():
            row[f"peso_{component}"] = round(float(value), 10)
        _append_csv(attempts_csv_path, row)

        attempts = _read_csv(attempts_csv_path)
        best_current = _best_for_target(attempts, target_concurso)
        state.update(
            {
                "last_attempt_finished_at": _now(),
                "last_attempt_elapsed_seconds": round(float(elapsed_attempt), 6),
                "last_hits_jogo_1": int(evaluation["acertos_jogo_1"]),
                "last_hits_jogo_2": int(evaluation["acertos_jogo_2"]),
                "last_best_hits": int(evaluation["melhor_acerto"]),
                "last_best_game": str(evaluation["melhor_jogo"]),
                "best_hits_current": int(best_current.get("hits", 0)),
                "best_game_current": str(best_current.get("game", "")),
                "best_attempt_current": int(best_current.get("attempt", 0)),
                "total_attempts": int(len(attempts)),
                "attempts_this_run": int(attempts_this_run),
                "elapsed_seconds_current_run": round(float(time.perf_counter() - started_perf), 6),
                "updated_at": _now(),
            }
        )

        if int(evaluation["melhor_acerto"]) >= 15:
            winner_row = dict(row)
            winner_row["solved_at"] = _now()
            _append_csv(winners_csv_path, winner_row)
            winners = _read_csv(winners_csv_path)
            average_winner_weights = _write_average_outputs(
                winners=winners,
                average_weights_csv_path=average_weights_csv_path,
                engine_weights_json_path=engine_weights_json_path,
            )
            solved = _solved_contests(winners)
            state.update(
                {
                    "status": "running",
                    "last_solved_concurso": int(target_concurso),
                    "last_solved_attempt": int(next_attempt),
                    "last_solved_at": _now(),
                    "solved_contests": sorted(solved),
                    "solved_count": int(len(solved)),
                    "average_weights": average_winner_weights or {},
                    "current_concurso": _next_target(contests, solved, None),
                    "current_attempt": 0,
                    "best_hits_current": 0,
                    "best_game_current": "",
                    "best_attempt_current": 0,
                    "updated_at": _now(),
                }
            )

        winners = _read_csv(winners_csv_path)
        _write_json(state_json_path, state)
        _write_summary(summary_csv_path=summary_csv_path, state=state, attempts=attempts, winners=winners)

        if attempts_this_run % 20 == 0 or int(evaluation["melhor_acerto"]) >= 15:
            _write_excel_snapshot(
                excel_path=excel_path,
                attempts=attempts,
                winners=winners,
                average_weights_csv_path=average_weights_csv_path,
                summary_csv_path=summary_csv_path,
            )

        if int(max_attempts) > 0 and attempts_this_run >= int(max_attempts):
            status = "paused_by_attempt_limit"
            state["status"] = status
            state["updated_at"] = _now()
            _write_json(state_json_path, state)
            break
        if int(max_runtime_seconds) > 0 and (time.perf_counter() - started_perf) >= int(max_runtime_seconds):
            status = "paused_by_runtime_limit"
            state["status"] = status
            state["updated_at"] = _now()
            _write_json(state_json_path, state)
            break

    attempts = _read_csv(attempts_csv_path)
    winners = _read_csv(winners_csv_path)
    best_current = _best_for_target(attempts, int(state["current_concurso"])) if state.get("current_concurso") else {"hits": 0, "game": ""}
    _write_average_outputs(
        winners=winners,
        average_weights_csv_path=average_weights_csv_path,
        engine_weights_json_path=engine_weights_json_path,
    )
    _write_summary(summary_csv_path=summary_csv_path, state=state, attempts=attempts, winners=winners)
    _write_excel_snapshot(
        excel_path=excel_path,
        attempts=attempts,
        winners=winners,
        average_weights_csv_path=average_weights_csv_path,
        summary_csv_path=summary_csv_path,
    )

    return CalibrationLabSummary(
        status=str(state.get("status", status)),
        current_concurso=int(state["current_concurso"]) if state.get("current_concurso") else None,
        attempts_this_run=int(attempts_this_run),
        total_attempts=int(len(attempts)),
        solved_contests=int(len(_solved_contests(winners))),
        best_hits_current=int(best_current.get("hits", 0)),
        best_game_current=str(best_current.get("game", "")),
        elapsed_seconds=round(float(time.perf_counter() - started_perf), 6),
        attempts_csv_path=str(attempts_csv_path),
        winners_csv_path=str(winners_csv_path),
        state_json_path=str(state_json_path),
        average_weights_csv_path=str(average_weights_csv_path),
        engine_weights_json_path=str(engine_weights_json_path),
    )
