from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import pandas as pd

from .calibrated_weights import load_supervised_calibrated_weights
from .climate_runtime import load_runtime_climate
from .config import AppConfig
from .exhaustive_optimizer import build_exhaustive_candidates, resolve_exhaustive_weights
from .normalize import DEZENAS
from .storage import load_processed_csv, sanitize_dataframe_for_tabular_output
from .top50_refinement import REFINEMENT_FEATURES, load_top50_refinement_payload, normalize_weights
from .top100_engine import (
    _actual_candidate_row,
    _build_coverage_hedge_candidates,
    _format_nums,
    _nums_from_row,
    enrich_candidates_with_top100_scores,
    select_top100_portfolio,
)
from .top100_learning import TOP100_WALKFORWARD_MODEL, apply_top100_learning, load_top100_learning_payload
from .top100_repair_learning import learn_from_prediction, load_top100_repair_payload


@dataclass(frozen=True)
class Top100WalkForwardLearningSummary:
    status: str
    contests_processed_this_run: int
    total_contests_processed: int
    current_concurso: int | None
    last_concurso: int | None
    best_hits_avg: float
    hit_top100: int
    weights_json_path: str
    state_json_path: str
    results_csv_path: str
    summary_csv_path: str
    weights_csv_path: str
    excel_path: str

    def to_console(self) -> str:
        return "\n".join(
            [
                "",
                "Resumo Lotofacil Analytics - Aprendizado Walk-forward Top100",
                f"Status: {self.status}",
                f"Concursos processados nesta execucao: {self.contests_processed_this_run}",
                f"Concursos processados no total: {self.total_contests_processed}",
                f"Concurso atual: {self.current_concurso if self.current_concurso is not None else 'nenhum'}",
                f"Ultimo concurso processado: {self.last_concurso if self.last_concurso is not None else 'nenhum'}",
                f"Melhor acerto medio entre os 100: {self.best_hits_avg:.4f}",
                f"Sequencia exata presente no Top100: {self.hit_top100}",
                f"Pesos Top100 aprendidos: {self.weights_json_path}",
                f"Estado retomavel: {self.state_json_path}",
                f"CSV resultados: {self.results_csv_path}",
                f"CSV resumo: {self.summary_csv_path}",
                f"CSV pesos: {self.weights_csv_path}",
                f"Excel: {self.excel_path}",
                "Mensagem: gera 100 jogos para concursos encerrados, compara com o gabarito e aprende para concursos futuros.",
            ]
        )


def _now() -> str:
    return pd.Timestamp.now().isoformat(timespec="seconds")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


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


def _target_climate_for(climate_features: pd.DataFrame | None, concurso: int) -> Mapping[str, object] | None:
    if climate_features is None or climate_features.empty or "concurso" not in climate_features.columns:
        return None
    df = climate_features.copy()
    df["concurso"] = pd.to_numeric(df["concurso"], errors="coerce")
    match = df[df["concurso"] == int(concurso)]
    return match.iloc[0].to_dict() if not match.empty else None


def _rank_for_actual(rows: pd.DataFrame, actual_text: str, score_column: str) -> Tuple[int, float]:
    if rows.empty or score_column not in rows.columns:
        return 0, 0.0
    ranked = rows.copy()
    ranked[score_column] = pd.to_numeric(ranked[score_column], errors="coerce").fillna(0.0)
    ranked = ranked.sort_values([score_column, "score_final", "nums"], ascending=[False, False, True]).reset_index(drop=True)
    matches = ranked.index[ranked["nums"] == actual_text].tolist()
    if not matches:
        return 0, 0.0
    rank = int(matches[0] + 1)
    percentile = round((1.0 - ((rank - 1) / max(1, len(ranked) - 1))) * 100.0, 6)
    return rank, percentile


def _feature_value(row: pd.Series, feature: str) -> float:
    try:
        value = float(row.get(feature, 50.0))
    except (TypeError, ValueError):
        value = 50.0
    return 50.0 if pd.isna(value) else max(0.0, min(100.0, value))


def _learn_top100_features(ranked: pd.DataFrame, *, actual_text: str, rank_before: int, best_hits: int) -> Dict[str, object]:
    actual_rows = ranked[ranked["nums"] == actual_text]
    if actual_rows.empty:
        return {"positive": {}, "negative": {}, "hard_negative_count": 0}
    actual = actual_rows.iloc[0]
    if int(rank_before) > 1:
        hard_negatives = ranked.iloc[: int(rank_before) - 1].head(500)
    else:
        hard_negatives = ranked[ranked["nums"] != actual_text].head(200)
    if hard_negatives.empty:
        return {"positive": {}, "negative": {}, "hard_negative_count": 0}

    positive: Dict[str, float] = {}
    negative: Dict[str, float] = {}
    severity = 1.0 + max(0, int(best_hits) - 10) * 0.65 + min(5.0, max(0.0, float(rank_before) - 100.0) / 80.0)
    for feature in REFINEMENT_FEATURES:
        if feature not in ranked.columns:
            continue
        actual_value = _feature_value(actual, feature)
        hard_values = pd.to_numeric(hard_negatives[feature], errors="coerce").fillna(50.0) if feature in hard_negatives.columns else pd.Series(dtype=float)
        if hard_values.empty:
            continue
        hard_avg = float(hard_values.mean())
        hard_p75 = float(hard_values.quantile(0.75))
        delta = actual_value - hard_avg
        if delta > 1.0:
            positive[feature] = positive.get(feature, 0.0) + delta * severity
        if delta < -1.0 and hard_p75 >= 54.0:
            negative[feature] = negative.get(feature, 0.0) + abs(delta) * severity
    return {"positive": positive, "negative": negative, "hard_negative_count": int(len(hard_negatives))}


def _average_payload_from_results(
    results: pd.DataFrame,
    *,
    from_concurso: int,
    to_concurso: int,
    min_history: int,
    top_count: int,
    top_pool: int,
    max_overlap: int,
    exhaustive_limit: int | None,
    draw_hour: int,
    draw_minute: int,
) -> Dict[str, object]:
    positive = {feature: 0.0 for feature in REFINEMENT_FEATURES}
    negative = {feature: 0.0 for feature in REFINEMENT_FEATURES}
    total_weight = 0.0
    if not results.empty:
        for _, row in results.iterrows():
            best_hits = float(row.get("best_hits_top100", 0) or 0)
            rank_before = float(row.get("rank_diagnostico_antes", 0) or 0)
            rank_after = float(row.get("rank_diagnostico_aprendido", rank_before) or rank_before)
            improvement = max(0.0, rank_before - rank_after)
            row_weight = 1.0 + max(0.0, best_hits - 9.0) * 0.65 + min(5.0, improvement / 80.0)
            total_weight += row_weight
            for feature in REFINEMENT_FEATURES:
                positive[feature] += float(row.get(f"pos_{feature}", 0.0) or 0.0) * row_weight
                negative[feature] += float(row.get(f"neg_{feature}", 0.0) or 0.0) * row_weight
    positive_norm = normalize_weights({feature: value / total_weight for feature, value in positive.items()}) if total_weight > 0 else {}
    negative_norm = normalize_weights({feature: value / total_weight for feature, value in negative.items()}) if total_weight > 0 else {}
    metrics: Dict[str, object] = {}
    if not results.empty:
        best_hits_series = pd.to_numeric(results["best_hits_top100"], errors="coerce").fillna(0)
        metrics = {
            "contests": int(results["concurso"].nunique()),
            "best_hits_avg": round(float(best_hits_series.mean()), 6),
            "best_hits_max": int(best_hits_series.max()),
            "hit_top100": int(pd.to_numeric(results["hit_top100"], errors="coerce").fillna(0).sum()),
            "hit_top100_pct": round(float(pd.to_numeric(results["hit_top100"], errors="coerce").fillna(0).mean()) * 100.0, 6),
            "rank_before_avg": round(float(pd.to_numeric(results["rank_diagnostico_antes"], errors="coerce").mean()), 6),
            "rank_after_avg": round(float(pd.to_numeric(results["rank_diagnostico_aprendido"], errors="coerce").mean()), 6),
        }
    return {
        "model": TOP100_WALKFORWARD_MODEL,
        "source": "historical_top100_walk_forward",
        "updated_at": _now(),
        "from_concurso": int(from_concurso),
        "to_concurso": int(to_concurso),
        "min_history": int(min_history),
        "top_count": int(top_count),
        "top_pool": int(top_pool),
        "max_overlap": int(max_overlap),
        "exhaustive_limit": int(exhaustive_limit) if exhaustive_limit is not None else None,
        "draw_hour": int(draw_hour),
        "draw_minute": int(draw_minute),
        "contests": int(results["concurso"].nunique()) if not results.empty else 0,
        "base_weight": 0.62,
        "positive_weights": positive_norm,
        "negative_weights": negative_norm,
        "metrics": metrics,
        "note": "Aprendido apenas depois de cada concurso historico; usado para reordenar concursos futuros.",
    }


def _write_outputs(
    *,
    results: pd.DataFrame,
    state: Mapping[str, object],
    summary_csv_path: Path,
    weights_csv_path: Path,
    weights_json_path: Path,
    excel_path: Path,
    from_concurso: int,
    to_concurso: int,
    min_history: int,
    top_count: int,
    top_pool: int,
    max_overlap: int,
    exhaustive_limit: int | None,
    draw_hour: int,
    draw_minute: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    payload = _average_payload_from_results(
        results,
        from_concurso=from_concurso,
        to_concurso=to_concurso,
        min_history=min_history,
        top_count=top_count,
        top_pool=top_pool,
        max_overlap=max_overlap,
        exhaustive_limit=exhaustive_limit,
        draw_hour=draw_hour,
        draw_minute=draw_minute,
    )
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
    summary = pd.DataFrame(
        [
            {"metrica": "status", "valor": state.get("status", "")},
            {"metrica": "concursos_processados", "valor": int(results["concurso"].nunique()) if not results.empty else 0},
            {"metrica": "melhor_acerto_medio_top100", "valor": metrics.get("best_hits_avg", 0.0)},
            {"metrica": "melhor_acerto_maximo_top100", "valor": metrics.get("best_hits_max", 0)},
            {"metrica": "hit_top100", "valor": metrics.get("hit_top100", 0)},
            {"metrica": "hit_top100_pct", "valor": metrics.get("hit_top100_pct", 0.0)},
            {"metrica": "rank_diagnostico_antes_medio", "valor": metrics.get("rank_before_avg", 0.0)},
            {"metrica": "rank_diagnostico_aprendido_medio", "valor": metrics.get("rank_after_avg", 0.0)},
            {"metrica": "top_count", "valor": int(top_count)},
            {"metrica": "top_pool", "valor": int(top_pool)},
            {"metrica": "exhaustive_limit", "valor": int(exhaustive_limit) if exhaustive_limit is not None else "completo"},
        ]
    )
    positive = payload.get("positive_weights", {}) if isinstance(payload.get("positive_weights"), dict) else {}
    negative = payload.get("negative_weights", {}) if isinstance(payload.get("negative_weights"), dict) else {}
    weight_rows = []
    for feature in REFINEMENT_FEATURES:
        weight_rows.append(
            {
                "feature": feature,
                "peso_positivo": round(float(positive.get(feature, 0.0)), 10),
                "peso_positivo_percentual": round(float(positive.get(feature, 0.0)) * 100.0, 6),
                "peso_penalizador": round(float(negative.get(feature, 0.0)), 10),
                "peso_penalizador_percentual": round(float(negative.get(feature, 0.0)) * 100.0, 6),
            }
        )
    weights_df = pd.DataFrame(weight_rows)
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    weights_csv_path.parent.mkdir(parents=True, exist_ok=True)
    weights_json_path.parent.mkdir(parents=True, exist_ok=True)
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    weights_df.to_csv(weights_csv_path, index=False, encoding="utf-8-sig")
    weights_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="resumo")
        weights_df.to_excel(writer, index=False, sheet_name="pesos_top100")
        results.tail(1000).to_excel(writer, index=False, sheet_name="resultados")
    return summary, payload


def load_top100_walkforward_status(
    *,
    state_json_path: Path,
    results_csv_path: Path,
    summary_csv_path: Path,
    weights_csv_path: Path,
    weights_json_path: Path,
    recent_rows: int = 12,
) -> Dict[str, object]:
    state = _load_json(state_json_path)
    results = _read_csv(results_csv_path)
    summary = _read_csv(summary_csv_path)
    weights = _read_csv(weights_csv_path)
    payload = load_top100_learning_payload(weights_json_path) or {}
    recent = results.tail(int(recent_rows)) if not results.empty else pd.DataFrame()
    blocks = pd.DataFrame()
    best = pd.DataFrame()
    if not results.empty:
        scored = results.copy()
        scored["concurso"] = pd.to_numeric(scored["concurso"], errors="coerce")
        scored["best_hits_top100"] = pd.to_numeric(scored["best_hits_top100"], errors="coerce")
        scored["rank_diagnostico_antes"] = pd.to_numeric(scored["rank_diagnostico_antes"], errors="coerce")
        scored["rank_diagnostico_aprendido"] = pd.to_numeric(scored["rank_diagnostico_aprendido"], errors="coerce")
        scored = scored.dropna(subset=["concurso"])
        state.setdefault("total_contests_processed", int(scored["concurso"].nunique()))
        state["best_hits_avg"] = round(float(scored["best_hits_top100"].mean()), 6)
        state["best_hits_max"] = int(scored["best_hits_top100"].max())
        state["hit_top100"] = int(pd.to_numeric(scored["hit_top100"], errors="coerce").fillna(0).sum())
        state["rank_before_avg"] = round(float(scored["rank_diagnostico_antes"].mean()), 6)
        state["rank_after_avg"] = round(float(scored["rank_diagnostico_aprendido"].mean()), 6)
        best = scored.sort_values(["best_hits_top100", "rank_diagnostico_aprendido", "concurso"], ascending=[False, True, False]).head(10)
        scored["bloco_inicio"] = (((scored["concurso"].astype(int) - 1) // 100) * 100 + 1).astype(int)
        scored["bloco_fim"] = scored["bloco_inicio"] + 99
        block_rows: List[Dict[str, object]] = []
        for (start, end), group in scored.groupby(["bloco_inicio", "bloco_fim"], sort=True):
            block_rows.append(
                {
                    "bloco": f"{int(start)}-{int(end)}",
                    "concursos": int(group["concurso"].nunique()),
                    "melhor_acerto_medio": round(float(group["best_hits_top100"].mean()), 6),
                    "melhor_acerto_maximo": int(group["best_hits_top100"].max()),
                    "rank_antes_medio": round(float(group["rank_diagnostico_antes"].mean()), 6),
                    "rank_aprendido_medio": round(float(group["rank_diagnostico_aprendido"].mean()), 6),
                    "hit_top100_pct": round(float(pd.to_numeric(group["hit_top100"], errors="coerce").fillna(0).mean()) * 100.0, 6),
                }
            )
        blocks = pd.DataFrame(block_rows)

    def records(df: pd.DataFrame) -> List[Dict[str, object]]:
        if df.empty:
            return []
        clean = df.astype(object).where(pd.notna(df), None)
        return clean.to_dict(orient="records")

    return {
        "state": state,
        "summary": records(summary),
        "weights": records(weights),
        "recent_results": records(recent),
        "best_results": records(best),
        "progress_blocks": records(blocks),
        "payload": payload,
        "paths": {
            "state_json_path": str(state_json_path),
            "results_csv_path": str(results_csv_path),
            "summary_csv_path": str(summary_csv_path),
            "weights_csv_path": str(weights_csv_path),
            "weights_json_path": str(weights_json_path),
        },
    }


class Top100WalkForwardLearningPipeline:
    def __init__(self, *, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def run(
        self,
        *,
        from_concurso: int,
        to_concurso: int | None,
        max_contests: int,
        min_history: int,
        top_count: int,
        top_pool: int,
        max_overlap: int,
        exhaustive_limit: int | None,
        seed: int,
        draw_hour: int,
        draw_minute: int,
        reset: bool = False,
    ) -> Top100WalkForwardLearningSummary:
        concursos = load_processed_csv(self.config.processed_csv_path)
        if concursos.empty:
            raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")
        climate_features, _target_climate = load_runtime_climate(
            config=self.config,
            concursos=concursos,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
        )
        return self._run_with_data(
            concursos=concursos,
            climate_features=climate_features,
            from_concurso=from_concurso,
            to_concurso=to_concurso,
            max_contests=max_contests,
            min_history=min_history,
            top_count=top_count,
            top_pool=top_pool,
            max_overlap=max_overlap,
            exhaustive_limit=exhaustive_limit,
            seed=seed,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
            reset=reset,
        )

    def _run_with_data(
        self,
        *,
        concursos: pd.DataFrame,
        climate_features: pd.DataFrame | None,
        from_concurso: int,
        to_concurso: int | None,
        max_contests: int,
        min_history: int,
        top_count: int,
        top_pool: int,
        max_overlap: int,
        exhaustive_limit: int | None,
        seed: int,
        draw_hour: int,
        draw_minute: int,
        reset: bool,
    ) -> Top100WalkForwardLearningSummary:
        if int(top_count) <= 0 or int(top_pool) <= 0:
            raise ValueError("top_count e top_pool devem ser maiores que zero.")
        if reset:
            for path in [
                self.config.top100_learning_state_json_path,
                self.config.top100_learning_results_csv_path,
                self.config.top100_learning_summary_csv_path,
                self.config.top100_learning_weights_csv_path,
                self.config.top100_learning_excel_path,
                self.config.top100_learning_weights_json_path,
            ]:
                if path.exists():
                    path.unlink()

        started = time.perf_counter()
        df = concursos.copy().sort_values("concurso").reset_index(drop=True)
        df["concurso"] = pd.to_numeric(df["concurso"], errors="coerce").astype("Int64")
        max_target = int(to_concurso) if to_concurso is not None else int(df["concurso"].max())
        contest_indices = {int(value): int(idx) for idx, value in df["concurso"].items() if pd.notna(value)}
        requested_targets = [
            int(value)
            for value in df["concurso"].dropna().astype(int).tolist()
            if int(value) >= int(from_concurso) and int(value) <= int(max_target)
        ]
        targets = [target for target in requested_targets if int(contest_indices.get(target, -1)) >= int(min_history)]
        skipped_min_history_count = len(requested_targets) - len(targets)
        if not targets:
            raise ValueError("Nenhum concurso elegivel no intervalo do aprendizado Top100.")

        results = _read_csv(self.config.top100_learning_results_csv_path)
        processed = set(int(value) for value in pd.to_numeric(results.get("concurso", pd.Series(dtype=int)), errors="coerce").dropna().tolist()) if not results.empty else set()
        target_set = set(targets)
        pending = sorted(target_set - processed)
        state = _load_json(self.config.top100_learning_state_json_path)
        state.update(
            {
                "status": "running",
                "started_last_run_at": _now(),
                "from_concurso": int(from_concurso),
                "to_concurso": int(max_target),
                "min_history": int(min_history),
                "top_count": int(top_count),
                "top_pool": int(top_pool),
                "max_overlap": int(max_overlap),
                "exhaustive_limit": int(exhaustive_limit) if exhaustive_limit is not None else None,
                "max_contests_this_run": int(max_contests),
                "seed": int(seed),
                "requested_target_count": int(len(requested_targets)),
                "eligible_target_count": int(len(targets)),
                "skipped_min_history_count": int(skipped_min_history_count),
                "processed_eligible_count": int(len(processed & target_set)),
                "remaining_eligible_count": int(len(pending)),
                "progress_percent": round(float(len(processed & target_set)) / float(len(targets)) * 100.0, 6),
                "next_pending_concurso": int(pending[0]) if pending else None,
                "draw_hour": int(draw_hour),
                "draw_minute": int(draw_minute),
            }
        )
        _write_json(self.config.top100_learning_state_json_path, state)

        base_weights = resolve_exhaustive_weights(load_supervised_calibrated_weights(self.config.supervised_calibration_weights_json_path))
        top50_payload = load_top50_refinement_payload(self.config.top50_refinement_weights_json_path)
        top100_payload = load_top100_learning_payload(self.config.top100_learning_weights_json_path)
        repair_payload = load_top100_repair_payload(self.config.top100_repair_weights_json_path) or {}
        processed_this_run = 0

        for target_concurso in targets:
            if target_concurso in processed:
                continue
            target_idx = int(contest_indices[target_concurso])
            if target_idx < int(min_history):
                continue
            train_df = df.iloc[:target_idx].copy()
            target_row = df.iloc[target_idx]
            actual_nums = _nums_from_row(target_row)
            actual_text = _format_nums(actual_nums)
            target_climate = _target_climate_for(climate_features, target_concurso)
            candidates, _summary = build_exhaustive_candidates(
                train_df,
                top_games=max(int(top_pool), int(top_count)),
                draw_hour=draw_hour,
                draw_minute=draw_minute,
                limit_combinations=exhaustive_limit,
                weights=base_weights,
                climate_features=climate_features,
                target_climate=target_climate,
            )
            base_pool = candidates.head(int(top_pool)).copy()
            hedge_pool = _build_coverage_hedge_candidates(
                train_df,
                base_pool["nums"].astype(str).tolist() if "nums" in base_pool.columns else [],
                top_count=int(top_count),
                target_rows=max(int(top_count) * 8, 800),
            )
            scored_pool = pd.concat([base_pool, hedge_pool], ignore_index=True).drop_duplicates(subset=["nums"], keep="first")
            enriched = enrich_candidates_with_top100_scores(
                scored_pool,
                train_df,
                refinement_payload=top50_payload,
                top100_learning_payload=top100_payload,
            )
            selected = select_top100_portfolio(enriched, top_count=int(top_count), max_overlap=int(max_overlap))
            selected_nums = set(str(value) for value in selected["nums"].tolist())
            actual_set = set(int(n) for n in actual_nums)
            best_hits = 0
            best_game = ""
            best_rank = 0
            for _, row in selected.iterrows():
                nums = set(int(part) for part in str(row["nums"]).split())
                hits = len(nums & actual_set)
                if hits > best_hits:
                    best_hits = int(hits)
                    best_game = str(row["nums"])
                    best_rank = int(row.get("rank_top100", 0) or 0)

            actual_row = _actual_candidate_row(
                train_df,
                target_row,
                climate_features=climate_features,
                draw_hour=draw_hour,
                draw_minute=draw_minute,
                weights=base_weights,
            )
            diagnostic_pool = pd.concat([enriched, actual_row], ignore_index=True)
            diagnostic = enrich_candidates_with_top100_scores(
                diagnostic_pool,
                train_df,
                refinement_payload=top50_payload,
                top100_learning_payload=top100_payload,
            ).drop_duplicates(subset=["nums"], keep="last")
            diagnostic = diagnostic.sort_values(["score_top100", "score_final", "nums"], ascending=[False, False, True]).reset_index(drop=True)
            rank_before, percentile_before = _rank_for_actual(diagnostic, actual_text, "score_top100")
            learned = _learn_top100_features(diagnostic, actual_text=actual_text, rank_before=rank_before, best_hits=best_hits)
            contest_payload = {
                "model": TOP100_WALKFORWARD_MODEL,
                "base_weight": 0.58,
                "positive_weights": normalize_weights(learned["positive"] if isinstance(learned["positive"], dict) else {}),
                "negative_weights": normalize_weights(learned["negative"] if isinstance(learned["negative"], dict) else {}),
            }
            learned_diagnostic = apply_top100_learning(diagnostic, contest_payload, override_score_top100=False)
            learned_diagnostic = learned_diagnostic.sort_values(["score_top100_aprendido", "score_top100", "nums"], ascending=[False, False, True]).reset_index(drop=True)
            rank_after, percentile_after = _rank_for_actual(learned_diagnostic, actual_text, "score_top100_aprendido")
            actual_learned = learned_diagnostic[learned_diagnostic["nums"] == actual_text].iloc[0]

            repair_payload, repair_rows = learn_from_prediction(
                selected,
                actual_numbers=actual_nums,
                concurso=target_concurso,
                payload=repair_payload,
                min_hits=11,
            )
            self.config.top100_repair_weights_json_path.parent.mkdir(parents=True, exist_ok=True)
            self.config.top100_repair_weights_json_path.write_text(json.dumps(repair_payload, ensure_ascii=False, indent=2), encoding="utf-8")

            result_row: Dict[str, object] = {
                "concurso": int(target_concurso),
                "data_sorteio": str(target_row.get("data_sorteio", "")),
                "processed_at": _now(),
                "jogo_real": actual_text,
                "top_count": int(top_count),
                "top_pool": int(top_pool),
                "selected_rows": int(len(selected)),
                "hit_top100": int(actual_text in selected_nums),
                "best_hits_top100": int(best_hits),
                "best_rank_top100": int(best_rank),
                "best_game_top100": best_game,
                "near_miss_11plus": int((pd.to_numeric(repair_rows["hits"], errors="coerce") >= 11).sum()) if not repair_rows.empty else 0,
                "rank_diagnostico_antes": int(rank_before),
                "rank_diagnostico_aprendido": int(rank_after),
                "melhora_rank_diagnostico": int(rank_before - rank_after),
                "percentil_antes": float(percentile_before),
                "percentil_aprendido": float(percentile_after),
                "score_top100_antes": round(float(actual_learned.get("score_top100", 0.0)), 6),
                "score_top100_aprendido": round(float(actual_learned.get("score_top100_aprendido", 0.0)), 6),
                "hard_negative_count": int(learned.get("hard_negative_count", 0)),
            }
            positive = learned["positive"] if isinstance(learned["positive"], dict) else {}
            negative = learned["negative"] if isinstance(learned["negative"], dict) else {}
            for feature in REFINEMENT_FEATURES:
                result_row[f"feature_real_{feature}"] = round(_feature_value(actual_learned, feature), 6)
                result_row[f"pos_{feature}"] = round(float(positive.get(feature, 0.0)), 6)
                result_row[f"neg_{feature}"] = round(float(negative.get(feature, 0.0)), 6)

            results = pd.concat([results, pd.DataFrame([result_row])], ignore_index=True)
            results = sanitize_dataframe_for_tabular_output(results)
            self.config.top100_learning_results_csv_path.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(self.config.top100_learning_results_csv_path, index=False, encoding="utf-8-sig")
            processed.add(int(target_concurso))
            processed_this_run += 1
            _summary_df, top100_payload = _write_outputs(
                results=results,
                state=state,
                summary_csv_path=self.config.top100_learning_summary_csv_path,
                weights_csv_path=self.config.top100_learning_weights_csv_path,
                weights_json_path=self.config.top100_learning_weights_json_path,
                excel_path=self.config.top100_learning_excel_path,
                from_concurso=from_concurso,
                to_concurso=max_target,
                min_history=min_history,
                top_count=top_count,
                top_pool=top_pool,
                max_overlap=max_overlap,
                exhaustive_limit=exhaustive_limit,
                draw_hour=draw_hour,
                draw_minute=draw_minute,
            )
            pending = sorted(target_set - processed)
            metrics = top100_payload.get("metrics", {}) if isinstance(top100_payload.get("metrics"), dict) else {}
            state.update(
                {
                    "status": "running",
                    "current_concurso": int(target_concurso),
                    "last_concurso": int(target_concurso),
                    "last_best_hits_top100": int(best_hits),
                    "last_best_game_top100": best_game,
                    "last_rank_diagnostico_antes": int(rank_before),
                    "last_rank_diagnostico_aprendido": int(rank_after),
                    "total_contests_processed": int(results["concurso"].nunique()),
                    "contests_processed_this_run": int(processed_this_run),
                    "processed_eligible_count": int(len(processed & target_set)),
                    "remaining_eligible_count": int(len(pending)),
                    "progress_percent": round(float(len(processed & target_set)) / float(len(targets)) * 100.0, 6),
                    "next_pending_concurso": int(pending[0]) if pending else None,
                    "best_hits_avg": metrics.get("best_hits_avg", 0.0),
                    "best_hits_max": metrics.get("best_hits_max", 0),
                    "hit_top100": metrics.get("hit_top100", 0),
                    "hit_top100_pct": metrics.get("hit_top100_pct", 0.0),
                    "rank_before_avg": metrics.get("rank_before_avg", 0.0),
                    "rank_after_avg": metrics.get("rank_after_avg", 0.0),
                    "updated_at": _now(),
                    "elapsed_seconds_current_run": round(float(time.perf_counter() - started), 6),
                }
            )
            _write_json(self.config.top100_learning_state_json_path, state)
            self.logger.info("Aprendizado Top100 processou concurso %s com melhor acerto %s", target_concurso, best_hits)
            if int(max_contests) > 0 and processed_this_run >= int(max_contests):
                state["status"] = "paused_by_contest_limit"
                state["updated_at"] = _now()
                _write_json(self.config.top100_learning_state_json_path, state)
                break

        pending = sorted(target_set - processed)
        if not pending:
            state["status"] = "complete"
        elif state.get("status") != "paused_by_contest_limit":
            state["status"] = "running"
        state["processed_eligible_count"] = int(len(processed & target_set))
        state["remaining_eligible_count"] = int(len(pending))
        state["progress_percent"] = round(float(len(processed & target_set)) / float(len(targets)) * 100.0, 6)
        state["next_pending_concurso"] = int(pending[0]) if pending else None
        state["updated_at"] = _now()
        state["elapsed_seconds_current_run"] = round(float(time.perf_counter() - started), 6)
        _write_json(self.config.top100_learning_state_json_path, state)
        _summary_df, payload = _write_outputs(
            results=results,
            state=state,
            summary_csv_path=self.config.top100_learning_summary_csv_path,
            weights_csv_path=self.config.top100_learning_weights_csv_path,
            weights_json_path=self.config.top100_learning_weights_json_path,
            excel_path=self.config.top100_learning_excel_path,
            from_concurso=from_concurso,
            to_concurso=max_target,
            min_history=min_history,
            top_count=top_count,
            top_pool=top_pool,
            max_overlap=max_overlap,
            exhaustive_limit=exhaustive_limit,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
        )
        metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
        return Top100WalkForwardLearningSummary(
            status=str(state.get("status", "")),
            contests_processed_this_run=int(processed_this_run),
            total_contests_processed=int(results["concurso"].nunique()) if not results.empty else 0,
            current_concurso=int(state["current_concurso"]) if state.get("current_concurso") else None,
            last_concurso=int(state["last_concurso"]) if state.get("last_concurso") else None,
            best_hits_avg=float(metrics.get("best_hits_avg", 0.0) or 0.0),
            hit_top100=int(metrics.get("hit_top100", 0) or 0),
            weights_json_path=str(self.config.top100_learning_weights_json_path),
            state_json_path=str(self.config.top100_learning_state_json_path),
            results_csv_path=str(self.config.top100_learning_results_csv_path),
            summary_csv_path=str(self.config.top100_learning_summary_csv_path),
            weights_csv_path=str(self.config.top100_learning_weights_csv_path),
            excel_path=str(self.config.top100_learning_excel_path),
        )

    def status(self) -> Dict[str, object]:
        return load_top100_walkforward_status(
            state_json_path=self.config.top100_learning_state_json_path,
            results_csv_path=self.config.top100_learning_results_csv_path,
            summary_csv_path=self.config.top100_learning_summary_csv_path,
            weights_csv_path=self.config.top100_learning_weights_csv_path,
            weights_json_path=self.config.top100_learning_weights_json_path,
        )
