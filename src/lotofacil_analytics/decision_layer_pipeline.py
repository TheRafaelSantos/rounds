from __future__ import annotations

import json
import logging

import pandas as pd

from .calibrated_weights import load_supervised_calibrated_weights
from .climate_runtime import load_runtime_climate
from .config import AppConfig
from .decision_layer import (
    ExhaustiveValidationSummary,
    SinglePredictionSummary,
    build_single_prediction,
    run_ablation_test,
    run_exhaustive_single_backtest,
    run_weight_tuning,
)
from .storage import load_processed_csv, sanitize_dataframe_for_tabular_output


class DecisionLayerPipeline:
    def __init__(self, *, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def predict_single(
        self,
        *,
        seed: int,
        candidate_pool: int,
        top_games: int,
        generations: int,
        population: int,
        draw_hour: int,
        draw_minute: int,
        engine: str,
        exhaustive_limit: int | None,
        weight_profile: str,
    ) -> SinglePredictionSummary:
        concursos = load_processed_csv(self.config.processed_csv_path)
        existing_candidates = load_processed_csv(self.config.optimizer_candidates_csv_path)
        climate_features, target_climate = load_runtime_climate(
            config=self.config,
            concursos=concursos,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
        )
        summary = build_single_prediction(
            concursos,
            existing_candidates=existing_candidates,
            seed=seed,
            candidate_pool=candidate_pool,
            top_games=top_games,
            generations=generations,
            population=population,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
            engine=engine,
            exhaustive_limit=exhaustive_limit,
            weight_profile=weight_profile,
            prediction_csv_path=self.config.single_prediction_csv_path,
            report_path=self.config.single_prediction_report_path,
            excel_path=self.config.single_prediction_excel_path,
            climate_features=climate_features,
            target_climate=target_climate,
            weights=load_supervised_calibrated_weights(self.config.supervised_calibration_weights_json_path),
        )
        self.logger.info("Jogo unico salvo em %s", self.config.single_prediction_csv_path)
        self.logger.info("Relatorio do jogo unico salvo em %s", self.config.single_prediction_report_path)
        return summary

    def backtest_exhaustive(
        self,
        *,
        n_eval: int,
        min_history: int,
        top_games: int,
        draw_hour: int,
        draw_minute: int,
        exhaustive_limit: int | None,
        weight_profile: str,
    ) -> ExhaustiveValidationSummary:
        concursos = load_processed_csv(self.config.processed_csv_path)
        climate_features, _target_climate = load_runtime_climate(
            config=self.config,
            concursos=concursos,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
        )
        results, summary = run_exhaustive_single_backtest(
            concursos,
            n_eval=n_eval,
            min_history=min_history,
            top_games=top_games,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
            exhaustive_limit=exhaustive_limit,
            weight_profile=weight_profile,
            climate_features=climate_features,
        )
        return self._save_validation_outputs(
            action="Backtest Exaustivo Jogo Unico",
            results=results,
            summary=summary,
            results_csv_path=self.config.exhaustive_backtest_csv_path,
            summary_csv_path=self.config.exhaustive_backtest_summary_csv_path,
            excel_path=self.config.exhaustive_backtest_excel_path,
            results_sheet="backtest",
            summary_sheet="resumo",
        )

    def ablation_test(
        self,
        *,
        n_eval: int,
        min_history: int,
        top_games: int,
        draw_hour: int,
        draw_minute: int,
        exhaustive_limit: int | None,
    ) -> ExhaustiveValidationSummary:
        concursos = load_processed_csv(self.config.processed_csv_path)
        climate_features, _target_climate = load_runtime_climate(
            config=self.config,
            concursos=concursos,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
        )
        results, summary = run_ablation_test(
            concursos,
            n_eval=n_eval,
            min_history=min_history,
            top_games=top_games,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
            exhaustive_limit=exhaustive_limit,
            climate_features=climate_features,
        )
        return self._save_validation_outputs(
            action="Ablation Test",
            results=results,
            summary=summary,
            results_csv_path=self.config.ablation_csv_path,
            summary_csv_path=self.config.ablation_summary_csv_path,
            excel_path=self.config.ablation_excel_path,
            results_sheet="ablation",
            summary_sheet="resumo",
        )

    def tune_weights(
        self,
        *,
        n_eval: int,
        min_history: int,
        top_games: int,
        draw_hour: int,
        draw_minute: int,
        exhaustive_limit: int | None,
    ) -> ExhaustiveValidationSummary:
        concursos = load_processed_csv(self.config.processed_csv_path)
        climate_features, _target_climate = load_runtime_climate(
            config=self.config,
            concursos=concursos,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
        )
        results, summary, best_payload = run_weight_tuning(
            concursos,
            n_eval=n_eval,
            min_history=min_history,
            top_games=top_games,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
            exhaustive_limit=exhaustive_limit,
            climate_features=climate_features,
        )
        self.config.tuned_weights_json_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.tuned_weights_json_path.write_text(
            json.dumps(best_payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return self._save_validation_outputs(
            action="Tune Weights",
            results=results,
            summary=summary,
            results_csv_path=self.config.tune_weights_csv_path,
            summary_csv_path=self.config.tune_weights_summary_csv_path,
            excel_path=self.config.tune_weights_excel_path,
            results_sheet="tuning",
            summary_sheet="resumo",
            extra_path=str(self.config.tuned_weights_json_path),
        )

    def _save_validation_outputs(
        self,
        *,
        action: str,
        results: pd.DataFrame,
        summary: pd.DataFrame,
        results_csv_path,
        summary_csv_path,
        excel_path,
        results_sheet: str,
        summary_sheet: str,
        extra_path: str | None = None,
    ) -> ExhaustiveValidationSummary:
        results = sanitize_dataframe_for_tabular_output(results)
        summary = sanitize_dataframe_for_tabular_output(summary)
        results_csv_path.parent.mkdir(parents=True, exist_ok=True)
        excel_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(results_csv_path, index=False, encoding="utf-8-sig")
        summary.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            results.to_excel(writer, index=False, sheet_name=results_sheet)
            summary.to_excel(writer, index=False, sheet_name=summary_sheet)

        self.logger.info("%s resultados salvos em %s", action, results_csv_path)
        self.logger.info("%s resumo salvo em %s", action, summary_csv_path)
        self.logger.info("%s Excel salvo em %s", action, excel_path)
        return self._validation_summary(
            action=action,
            results=results,
            summary=summary,
            results_csv_path=str(results_csv_path),
            summary_csv_path=str(summary_csv_path),
            excel_path=str(excel_path),
            extra_path=extra_path,
        )

    def _validation_summary(
        self,
        *,
        action: str,
        results: pd.DataFrame,
        summary: pd.DataFrame,
        results_csv_path: str,
        summary_csv_path: str,
        excel_path: str,
        extra_path: str | None = None,
    ) -> ExhaustiveValidationSummary:
        if results.empty:
            raise ValueError(f"{action} nao gerou resultados.")
        if summary.empty:
            raise ValueError(f"{action} nao gerou resumo.")
        best = summary.iloc[0]
        return ExhaustiveValidationSummary(
            action=action,
            rows=int(len(results)),
            contests=int(results["concurso_previsto"].nunique()),
            first_concurso=int(results["concurso_previsto"].min()),
            last_concurso=int(results["concurso_previsto"].max()),
            average_hits=float(best["media_acertos_jogo_unico"]),
            best_hits=int(best["max_acertos"]),
            best_profile=str(best["weight_profile"]),
            results_csv_path=results_csv_path,
            summary_csv_path=summary_csv_path,
            excel_path=excel_path,
            extra_path=extra_path,
        )
