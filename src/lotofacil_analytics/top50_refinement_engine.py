from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import pandas as pd

from .exhaustive_optimizer import build_exhaustive_candidates, resolve_exhaustive_weights
from .normalize import DEZENAS
from .storage import sanitize_dataframe_for_tabular_output
from .supervised_calibration import _apply_weights, _score_candidate_table
from .top100_engine import enrich_candidates_with_top100_scores
from .top50_refinement import (
    REFINEMENT_FEATURES,
    TOP50_REFINEMENT_MODEL,
    apply_top50_refinement,
    normalize_weights,
)


@dataclass(frozen=True)
class Top50RefinementSummary:
    status: str
    contests_processed_this_run: int
    total_contests_processed: int
    current_concurso: int | None
    last_concurso: int | None
    rank_before_avg: float
    rank_after_avg: float
    hit_top50_before: float
    hit_top50_after: float
    hit_top100_before: float
    hit_top100_after: float
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
                "Resumo Lotofacil Analytics - Motor 3.0 Refinador Top50",
                f"Status: {self.status}",
                f"Concursos processados nesta execucao: {self.contests_processed_this_run}",
                f"Concursos processados no total: {self.total_contests_processed}",
                f"Concurso atual: {self.current_concurso if self.current_concurso is not None else 'nenhum'}",
                f"Ultimo concurso processado: {self.last_concurso if self.last_concurso is not None else 'nenhum'}",
                f"Rank medio antes: {self.rank_before_avg:.4f}",
                f"Rank medio depois: {self.rank_after_avg:.4f}",
                f"Hit@50 antes/depois: {self.hit_top50_before:.2f}% / {self.hit_top50_after:.2f}%",
                f"Hit@100 antes/depois: {self.hit_top100_before:.2f}% / {self.hit_top100_after:.2f}%",
                f"Pesos refinados: {self.weights_json_path}",
                f"Estado retomavel: {self.state_json_path}",
                f"CSV resultados: {self.results_csv_path}",
                f"CSV resumo: {self.summary_csv_path}",
                f"CSV pesos: {self.weights_csv_path}",
                f"Excel: {self.excel_path}",
                "Mensagem: aprende com erros historicos ja encerrados para reordenar concursos futuros, sem usar gabarito futuro.",
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


def _nums_from_row(row: pd.Series) -> Tuple[int, ...]:
    return tuple(sorted(int(row[col]) for col in DEZENAS))


def _format_nums(nums: Sequence[int]) -> str:
    return " ".join(f"{int(n):02d}" for n in sorted(nums))


def _rank_for_actual(rows: pd.DataFrame, actual_text: str, score_column: str) -> Tuple[int, float]:
    if rows.empty or score_column not in rows.columns:
        return 0, 0.0
    ranked = rows.copy()
    ranked[score_column] = pd.to_numeric(ranked[score_column], errors="coerce").fillna(0.0)
    ranked = ranked.sort_values([score_column, "nums"], ascending=[False, True]).reset_index(drop=True)
    matches = ranked.index[ranked["nums"] == actual_text].tolist()
    if not matches:
        return 0, 0.0
    rank = int(matches[0] + 1)
    percentile = round((1.0 - ((rank - 1) / max(1, len(ranked) - 1))) * 100.0, 6)
    return rank, percentile


def _actual_candidate_row(
    train_df: pd.DataFrame,
    target_row: pd.Series,
    *,
    climate_features: pd.DataFrame | None,
    draw_hour: int,
    draw_minute: int,
    weights: Mapping[str, float],
) -> pd.DataFrame:
    actual = _nums_from_row(target_row)
    scored = _score_candidate_table(
        train_df=train_df,
        target_row=target_row,
        candidates=[actual],
        climate_features=climate_features,
        draw_hour=draw_hour,
        draw_minute=draw_minute,
    )
    scored = _apply_weights(scored, weights, column="score_final")
    scored["source_model"] = "gabarito_historico_refinador_top50"
    scored["metodo"] = "gabarito_historico_refinador_top50"
    return scored


def _target_climate_for(climate_features: pd.DataFrame | None, concurso: int) -> Mapping[str, object] | None:
    if climate_features is None or climate_features.empty or "concurso" not in climate_features.columns:
        return None
    df = climate_features.copy()
    df["concurso"] = pd.to_numeric(df["concurso"], errors="coerce")
    match = df[df["concurso"] == int(concurso)]
    return match.iloc[0].to_dict() if not match.empty else None


def _feature_value(row: pd.Series, feature: str) -> float:
    try:
        value = float(row.get(feature, 50.0))
    except (TypeError, ValueError):
        value = 50.0
    return 50.0 if pd.isna(value) else max(0.0, min(100.0, value))


def _learn_refinement_for_contest(
    ranked: pd.DataFrame,
    *,
    actual_text: str,
    rank_before: int,
) -> Dict[str, object]:
    actual_rows = ranked[ranked["nums"] == actual_text]
    if actual_rows.empty:
        return {"positive": {}, "negative": {}, "hard_negative_count": 0}
    actual = actual_rows.iloc[0]
    if int(rank_before) > 1:
        hard_negatives = ranked.iloc[: int(rank_before) - 1].head(300)
    else:
        hard_negatives = ranked.head(100)
        hard_negatives = hard_negatives[hard_negatives["nums"] != actual_text]
    if hard_negatives.empty:
        hard_negatives = ranked[ranked["nums"] != actual_text].head(100)

    positive_raw: Dict[str, float] = {}
    negative_raw: Dict[str, float] = {}
    severity = min(5.0, 1.0 + max(0.0, float(rank_before) - 50.0) / 50.0)
    for feature in REFINEMENT_FEATURES:
        if feature not in ranked.columns:
            continue
        actual_value = _feature_value(actual, feature)
        hard_values = pd.to_numeric(hard_negatives[feature], errors="coerce").fillna(50.0) if feature in hard_negatives.columns else pd.Series(dtype=float)
        if hard_values.empty:
            continue
        hard_avg = float(hard_values.mean())
        hard_p80 = float(hard_values.quantile(0.80))
        delta = actual_value - hard_avg
        if delta > 1.25:
            positive_raw[feature] = positive_raw.get(feature, 0.0) + (delta * severity)
        if delta < -1.25 and hard_p80 >= 55.0:
            negative_raw[feature] = negative_raw.get(feature, 0.0) + (abs(delta) * severity)
    return {
        "positive": positive_raw,
        "negative": negative_raw,
        "hard_negative_count": int(len(hard_negatives)),
    }


def _average_refinement_weights(results: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
    if results.empty:
        return {}, {}
    positive = {feature: 0.0 for feature in REFINEMENT_FEATURES}
    negative = {feature: 0.0 for feature in REFINEMENT_FEATURES}
    total_weight = 0.0
    for _, row in results.iterrows():
        rank_before = float(row.get("rank_top100_antes", 0) or 0)
        rank_after = float(row.get("rank_top50_refinado", rank_before) or rank_before)
        improvement = max(0.0, rank_before - rank_after)
        severity = 1.0 + max(0.0, rank_before - 50.0) / 100.0 + improvement / 100.0
        total_weight += severity
        for feature in REFINEMENT_FEATURES:
            positive[feature] += float(row.get(f"pos_{feature}", 0.0) or 0.0) * severity
            negative[feature] += float(row.get(f"neg_{feature}", 0.0) or 0.0) * severity
    if total_weight <= 0.0:
        return {}, {}
    return (
        normalize_weights({feature: value / total_weight for feature, value in positive.items()}),
        normalize_weights({feature: value / total_weight for feature, value in negative.items()}),
    )


def _payload_from_results(
    results: pd.DataFrame,
    *,
    from_concurso: int,
    to_concurso: int,
    min_history: int,
    top_pool: int,
    exhaustive_limit: int | None,
    draw_hour: int,
    draw_minute: int,
) -> Dict[str, object]:
    positive, negative = _average_refinement_weights(results)
    base_weight = 0.56
    metrics: Dict[str, object] = {}
    if not results.empty:
        metrics = {
            "rank_before_avg": round(float(pd.to_numeric(results["rank_top100_antes"], errors="coerce").mean()), 6),
            "rank_after_avg": round(float(pd.to_numeric(results["rank_top50_refinado"], errors="coerce").mean()), 6),
            "hit_top50_before": round(float(pd.to_numeric(results["hit_top50_antes"], errors="coerce").fillna(0).mean()) * 100.0, 6),
            "hit_top50_after": round(float(pd.to_numeric(results["hit_top50_refinado"], errors="coerce").fillna(0).mean()) * 100.0, 6),
            "hit_top100_before": round(float(pd.to_numeric(results["hit_top100_antes"], errors="coerce").fillna(0).mean()) * 100.0, 6),
            "hit_top100_after": round(float(pd.to_numeric(results["hit_top100_refinado"], errors="coerce").fillna(0).mean()) * 100.0, 6),
        }
    return {
        "model": TOP50_REFINEMENT_MODEL,
        "source": "historical_post_error_refinement_walk_forward",
        "updated_at": _now(),
        "from_concurso": int(from_concurso),
        "to_concurso": int(to_concurso),
        "min_history": int(min_history),
        "top_pool": int(top_pool),
        "exhaustive_limit": int(exhaustive_limit) if exhaustive_limit is not None else None,
        "draw_hour": int(draw_hour),
        "draw_minute": int(draw_minute),
        "contests": int(results["concurso"].nunique()) if not results.empty else 0,
        "base_weight": base_weight,
        "positive_weights": positive,
        "negative_weights": negative,
        "metrics": metrics,
        "note": "Pesos aprendidos apos concursos historicos ja encerrados; usados somente para reordenar concursos futuros.",
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
    top_pool: int,
    exhaustive_limit: int | None,
    draw_hour: int,
    draw_minute: int,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    payload = _payload_from_results(
        results,
        from_concurso=from_concurso,
        to_concurso=to_concurso,
        min_history=min_history,
        top_pool=top_pool,
        exhaustive_limit=exhaustive_limit,
        draw_hour=draw_hour,
        draw_minute=draw_minute,
    )
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
    summary = pd.DataFrame(
        [
            {"metrica": "status", "valor": state.get("status", "")},
            {"metrica": "concursos_processados", "valor": int(results["concurso"].nunique()) if not results.empty else 0},
            {"metrica": "rank_antes_medio", "valor": metrics.get("rank_before_avg", 0.0)},
            {"metrica": "rank_refinado_medio", "valor": metrics.get("rank_after_avg", 0.0)},
            {"metrica": "hit_top50_antes_pct", "valor": metrics.get("hit_top50_before", 0.0)},
            {"metrica": "hit_top50_refinado_pct", "valor": metrics.get("hit_top50_after", 0.0)},
            {"metrica": "hit_top100_antes_pct", "valor": metrics.get("hit_top100_before", 0.0)},
            {"metrica": "hit_top100_refinado_pct", "valor": metrics.get("hit_top100_after", 0.0)},
            {"metrica": "top_pool", "valor": int(top_pool)},
            {"metrica": "exhaustive_limit", "valor": int(exhaustive_limit) if exhaustive_limit is not None else "completo"},
        ]
    )
    positive = payload.get("positive_weights", {}) if isinstance(payload.get("positive_weights"), dict) else {}
    negative = payload.get("negative_weights", {}) if isinstance(payload.get("negative_weights"), dict) else {}
    weights_rows = []
    for feature in REFINEMENT_FEATURES:
        weights_rows.append(
            {
                "feature": feature,
                "peso_positivo": round(float(positive.get(feature, 0.0)), 10),
                "peso_positivo_percentual": round(float(positive.get(feature, 0.0)) * 100.0, 6),
                "peso_penalizador": round(float(negative.get(feature, 0.0)), 10),
                "peso_penalizador_percentual": round(float(negative.get(feature, 0.0)) * 100.0, 6),
            }
        )
    weights_df = pd.DataFrame(weights_rows)
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    weights_csv_path.parent.mkdir(parents=True, exist_ok=True)
    weights_json_path.parent.mkdir(parents=True, exist_ok=True)
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    weights_df.to_csv(weights_csv_path, index=False, encoding="utf-8-sig")
    weights_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="resumo")
        weights_df.to_excel(writer, index=False, sheet_name="pesos_refinados")
        results.tail(1000).to_excel(writer, index=False, sheet_name="resultados")
    return summary, payload


def run_top50_refinement(
    concursos: pd.DataFrame,
    *,
    climate_features: pd.DataFrame | None,
    from_concurso: int,
    to_concurso: int | None,
    max_contests: int,
    min_history: int,
    top_pool: int,
    exhaustive_limit: int | None,
    seed: int,
    draw_hour: int,
    draw_minute: int,
    reset: bool,
    base_weights: Mapping[str, float] | None,
    state_json_path: Path,
    results_csv_path: Path,
    summary_csv_path: Path,
    weights_csv_path: Path,
    excel_path: Path,
    weights_json_path: Path,
) -> Top50RefinementSummary:
    if concursos.empty:
        raise ValueError("Historico local nao encontrado.")
    if int(top_pool) <= 0:
        raise ValueError("--top50-refine-pool deve ser maior que zero.")
    if reset:
        for path in [state_json_path, results_csv_path, summary_csv_path, weights_csv_path, excel_path, weights_json_path]:
            if path.exists():
                path.unlink()

    started = time.perf_counter()
    resolved_base_weights = resolve_exhaustive_weights(base_weights)
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
    if not targets:
        raise ValueError("Nenhum concurso elegivel no intervalo do refinador Top50.")

    results = _read_csv(results_csv_path)
    processed = set(int(value) for value in pd.to_numeric(results.get("concurso", pd.Series(dtype=int)), errors="coerce").dropna().tolist()) if not results.empty else set()
    target_set = set(targets)
    pending = sorted(target_set - processed)
    state = _load_json(state_json_path)
    state.update(
        {
            "status": "running",
            "started_last_run_at": _now(),
            "from_concurso": int(from_concurso),
            "to_concurso": int(max_target),
            "min_history": int(min_history),
            "top_pool": int(top_pool),
            "exhaustive_limit": int(exhaustive_limit) if exhaustive_limit is not None else None,
            "max_contests_this_run": int(max_contests),
            "seed": int(seed),
            "eligible_target_count": int(len(targets)),
            "processed_eligible_count": int(len(processed & target_set)),
            "remaining_eligible_count": int(len(pending)),
            "progress_percent": round(float(len(processed & target_set)) / float(len(targets)) * 100.0, 6),
            "next_pending_concurso": int(pending[0]) if pending else None,
            "draw_hour": int(draw_hour),
            "draw_minute": int(draw_minute),
        }
    )
    _write_json(state_json_path, state)

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
            top_games=int(top_pool),
            draw_hour=draw_hour,
            draw_minute=draw_minute,
            limit_combinations=exhaustive_limit,
            weights=resolved_base_weights,
            climate_features=climate_features,
            target_climate=target_climate,
        )
        base_enriched = enrich_candidates_with_top100_scores(candidates.head(int(top_pool)), train_df, refinement_payload=None)
        actual_row = _actual_candidate_row(
            train_df,
            target_row,
            climate_features=climate_features,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
            weights=resolved_base_weights,
        )
        diagnostic_pool = pd.concat([base_enriched, actual_row], ignore_index=True)
        diagnostic = enrich_candidates_with_top100_scores(diagnostic_pool, train_df, refinement_payload=None)
        diagnostic = diagnostic.drop_duplicates(subset=["nums"], keep="last")
        diagnostic = diagnostic.sort_values(["score_top100", "score_final", "nums"], ascending=[False, False, True]).reset_index(drop=True)
        rank_before, percentile_before = _rank_for_actual(diagnostic, actual_text, "score_top100")
        learned = _learn_refinement_for_contest(diagnostic, actual_text=actual_text, rank_before=rank_before)
        contest_payload = {
            "model": TOP50_REFINEMENT_MODEL,
            "base_weight": 0.56,
            "positive_weights": normalize_weights(learned["positive"] if isinstance(learned["positive"], dict) else {}),
            "negative_weights": normalize_weights(learned["negative"] if isinstance(learned["negative"], dict) else {}),
        }
        refined = apply_top50_refinement(diagnostic, contest_payload, override_score_top100=False)
        refined = refined.sort_values(["score_top50_refinado", "score_top100", "nums"], ascending=[False, False, True]).reset_index(drop=True)
        rank_after, percentile_after = _rank_for_actual(refined, actual_text, "score_top50_refinado")
        actual_refined = refined[refined["nums"] == actual_text].iloc[0]
        result_row: Dict[str, object] = {
            "concurso": int(target_concurso),
            "data_sorteio": str(target_row.get("data_sorteio", "")),
            "processed_at": _now(),
            "jogo_real": actual_text,
            "top_pool": int(top_pool),
            "rank_top100_antes": int(rank_before),
            "rank_top50_refinado": int(rank_after),
            "melhora_rank_refinador": int(rank_before - rank_after),
            "hit_top50_antes": int(0 < int(rank_before) <= 50),
            "hit_top50_refinado": int(0 < int(rank_after) <= 50),
            "hit_top100_antes": int(0 < int(rank_before) <= 100),
            "hit_top100_refinado": int(0 < int(rank_after) <= 100),
            "percentil_antes": float(percentile_before),
            "percentil_refinado": float(percentile_after),
            "score_top100_antes": round(float(actual_refined.get("score_top100", 0.0)), 6),
            "score_top50_refinado": round(float(actual_refined.get("score_top50_refinado", 0.0)), 6),
            "hard_negative_count": int(learned.get("hard_negative_count", 0)),
        }
        positive = learned["positive"] if isinstance(learned["positive"], dict) else {}
        negative = learned["negative"] if isinstance(learned["negative"], dict) else {}
        for feature in REFINEMENT_FEATURES:
            result_row[f"feature_real_{feature}"] = round(_feature_value(actual_refined, feature), 6)
            result_row[f"pos_{feature}"] = round(float(positive.get(feature, 0.0)), 6)
            result_row[f"neg_{feature}"] = round(float(negative.get(feature, 0.0)), 6)

        results = pd.concat([results, pd.DataFrame([result_row])], ignore_index=True)
        results = sanitize_dataframe_for_tabular_output(results)
        results_csv_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(results_csv_path, index=False, encoding="utf-8-sig")
        processed.add(int(target_concurso))
        processed_this_run += 1
        summary_df, payload = _write_outputs(
            results=results,
            state=state,
            summary_csv_path=summary_csv_path,
            weights_csv_path=weights_csv_path,
            weights_json_path=weights_json_path,
            excel_path=excel_path,
            from_concurso=from_concurso,
            to_concurso=max_target,
            min_history=min_history,
            top_pool=top_pool,
            exhaustive_limit=exhaustive_limit,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
        )
        pending = sorted(target_set - processed)
        metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
        state.update(
            {
                "status": "running",
                "current_concurso": int(target_concurso),
                "last_concurso": int(target_concurso),
                "last_rank_top100_antes": int(rank_before),
                "last_rank_top50_refinado": int(rank_after),
                "last_melhora_rank_refinador": int(rank_before - rank_after),
                "total_contests_processed": int(results["concurso"].nunique()),
                "contests_processed_this_run": int(processed_this_run),
                "processed_eligible_count": int(len(processed & target_set)),
                "remaining_eligible_count": int(len(pending)),
                "progress_percent": round(float(len(processed & target_set)) / float(len(targets)) * 100.0, 6),
                "next_pending_concurso": int(pending[0]) if pending else None,
                "rank_before_avg": metrics.get("rank_before_avg", 0.0),
                "rank_after_avg": metrics.get("rank_after_avg", 0.0),
                "hit_top50_before": metrics.get("hit_top50_before", 0.0),
                "hit_top50_after": metrics.get("hit_top50_after", 0.0),
                "hit_top100_before": metrics.get("hit_top100_before", 0.0),
                "hit_top100_after": metrics.get("hit_top100_after", 0.0),
                "updated_at": _now(),
                "elapsed_seconds_current_run": round(float(time.perf_counter() - started), 6),
            }
        )
        _write_json(state_json_path, state)
        if int(max_contests) > 0 and processed_this_run >= int(max_contests):
            state["status"] = "paused_by_contest_limit"
            state["updated_at"] = _now()
            _write_json(state_json_path, state)
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
    _write_json(state_json_path, state)
    _summary_df, payload = _write_outputs(
        results=results,
        state=state,
        summary_csv_path=summary_csv_path,
        weights_csv_path=weights_csv_path,
        weights_json_path=weights_json_path,
        excel_path=excel_path,
        from_concurso=from_concurso,
        to_concurso=max_target,
        min_history=min_history,
        top_pool=top_pool,
        exhaustive_limit=exhaustive_limit,
        draw_hour=draw_hour,
        draw_minute=draw_minute,
    )
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
    return Top50RefinementSummary(
        status=str(state.get("status", "")),
        contests_processed_this_run=int(processed_this_run),
        total_contests_processed=int(results["concurso"].nunique()) if not results.empty else 0,
        current_concurso=int(state["current_concurso"]) if state.get("current_concurso") else None,
        last_concurso=int(state["last_concurso"]) if state.get("last_concurso") else None,
        rank_before_avg=float(metrics.get("rank_before_avg", 0.0) or 0.0),
        rank_after_avg=float(metrics.get("rank_after_avg", 0.0) or 0.0),
        hit_top50_before=float(metrics.get("hit_top50_before", 0.0) or 0.0),
        hit_top50_after=float(metrics.get("hit_top50_after", 0.0) or 0.0),
        hit_top100_before=float(metrics.get("hit_top100_before", 0.0) or 0.0),
        hit_top100_after=float(metrics.get("hit_top100_after", 0.0) or 0.0),
        weights_json_path=str(weights_json_path),
        state_json_path=str(state_json_path),
        results_csv_path=str(results_csv_path),
        summary_csv_path=str(summary_csv_path),
        weights_csv_path=str(weights_csv_path),
        excel_path=str(excel_path),
    )


def load_top50_refinement_status(
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
    payload = _load_json(weights_json_path)
    recent = results.tail(int(recent_rows)) if not results.empty else pd.DataFrame()
    blocks = pd.DataFrame()
    best = pd.DataFrame()
    if not results.empty:
        scored = results.copy()
        scored["concurso"] = pd.to_numeric(scored["concurso"], errors="coerce")
        scored["rank_top100_antes"] = pd.to_numeric(scored["rank_top100_antes"], errors="coerce")
        scored["rank_top50_refinado"] = pd.to_numeric(scored["rank_top50_refinado"], errors="coerce")
        scored["melhora_rank_refinador"] = pd.to_numeric(scored["melhora_rank_refinador"], errors="coerce")
        scored = scored.dropna(subset=["concurso"])
        state.setdefault("total_contests_processed", int(scored["concurso"].nunique()))
        state.setdefault("rank_before_avg", round(float(scored["rank_top100_antes"].mean()), 6))
        state.setdefault("rank_after_avg", round(float(scored["rank_top50_refinado"].mean()), 6))
        state["rank_improvement_avg"] = round(float(scored["melhora_rank_refinador"].mean()), 6)
        state["hit_top50_before"] = round(float(pd.to_numeric(scored["hit_top50_antes"], errors="coerce").fillna(0).mean()) * 100.0, 6)
        state["hit_top50_after"] = round(float(pd.to_numeric(scored["hit_top50_refinado"], errors="coerce").fillna(0).mean()) * 100.0, 6)
        state["hit_top100_before"] = round(float(pd.to_numeric(scored["hit_top100_antes"], errors="coerce").fillna(0).mean()) * 100.0, 6)
        state["hit_top100_after"] = round(float(pd.to_numeric(scored["hit_top100_refinado"], errors="coerce").fillna(0).mean()) * 100.0, 6)
        best = scored.sort_values(["rank_top50_refinado", "rank_top100_antes", "concurso"], ascending=[True, True, False]).head(10)
        scored["bloco_inicio"] = (((scored["concurso"].astype(int) - 1) // 100) * 100 + 1).astype(int)
        scored["bloco_fim"] = scored["bloco_inicio"] + 99
        block_rows: List[Dict[str, object]] = []
        for (start, end), group in scored.groupby(["bloco_inicio", "bloco_fim"], sort=True):
            block_rows.append(
                {
                    "bloco": f"{int(start)}-{int(end)}",
                    "concursos": int(group["concurso"].nunique()),
                    "rank_antes_medio": round(float(group["rank_top100_antes"].mean()), 6),
                    "rank_refinado_medio": round(float(group["rank_top50_refinado"].mean()), 6),
                    "melhora_media": round(float(group["melhora_rank_refinador"].mean()), 6),
                    "hit_top50_refinado_pct": round(float(pd.to_numeric(group["hit_top50_refinado"], errors="coerce").fillna(0).mean()) * 100.0, 6),
                    "hit_top100_refinado_pct": round(float(pd.to_numeric(group["hit_top100_refinado"], errors="coerce").fillna(0).mean()) * 100.0, 6),
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
