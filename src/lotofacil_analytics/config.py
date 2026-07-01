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

    @property
    def dezenas_long_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_dezenas_long.csv"

    @property
    def dezenas_historico_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_dezenas_historico.csv"

    @property
    def dezenas_historico_excel_path(self) -> Path:
        return self.exports_dir / "lotofacil_dezenas_historico.xlsx"

    @property
    def combinacoes_features_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_combinacoes_features.csv"

    @property
    def combinacoes_pares_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_combinacoes_pares.csv"

    @property
    def combinacoes_trios_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_combinacoes_trios.csv"

    @property
    def combinacoes_quartetos_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_combinacoes_quartetos.csv"

    @property
    def combinacoes_excel_path(self) -> Path:
        return self.exports_dir / "lotofacil_combinacoes.xlsx"

    @property
    def transition_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_transicoes.csv"

    @property
    def transition_summary_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_transicoes_summary.csv"

    @property
    def transition_dezenas_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_transicoes_dezenas.csv"

    @property
    def transition_excel_path(self) -> Path:
        return self.exports_dir / "lotofacil_transicoes.xlsx"

    @property
    def climate_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_clima.csv"

    @property
    def climate_summary_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_clima_summary.csv"

    @property
    def climate_excel_path(self) -> Path:
        return self.exports_dir / "lotofacil_clima.xlsx"

    @property
    def climate_cache_dir(self) -> Path:
        return self.data_dir / "raw" / "climate_open_meteo"

    @property
    def temporal_deep_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_temporal_profundo.csv"

    @property
    def temporal_deep_summary_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_temporal_profundo_summary.csv"

    @property
    def temporal_deep_excel_path(self) -> Path:
        return self.exports_dir / "lotofacil_temporal_profundo.xlsx"

    @property
    def engine_calibration_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_engine_calibration.csv"

    @property
    def engine_calibration_summary_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_engine_calibration_summary.csv"

    @property
    def engine_calibration_weights_json_path(self) -> Path:
        return self.processed_dir / "lotofacil_engine_calibrated_weights.json"

    @property
    def engine_calibration_excel_path(self) -> Path:
        return self.exports_dir / "lotofacil_engine_calibration.xlsx"

    @property
    def backtest_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_backtest.csv"

    @property
    def backtest_summary_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_backtest_summary.csv"

    @property
    def backtest_excel_path(self) -> Path:
        return self.exports_dir / "lotofacil_backtest.xlsx"

    @property
    def auditoria_resumo_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_auditoria_resumo.csv"

    @property
    def auditoria_dezenas_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_auditoria_dezenas.csv"

    @property
    def auditoria_anomalias_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_auditoria_anomalias.csv"

    @property
    def auditoria_monte_carlo_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_auditoria_monte_carlo.csv"

    @property
    def auditoria_excel_path(self) -> Path:
        return self.exports_dir / "lotofacil_auditoria.xlsx"

    @property
    def ml_dataset_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_ml_dataset.csv"

    @property
    def ml_predictions_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_ml_predictions.csv"

    @property
    def ml_summary_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_ml_summary.csv"

    @property
    def ml_excel_path(self) -> Path:
        return self.exports_dir / "lotofacil_ml.xlsx"

    @property
    def optimizer_candidates_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_optimizer_candidates.csv"

    @property
    def optimizer_summary_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_optimizer_summary.csv"

    @property
    def optimizer_excel_path(self) -> Path:
        return self.exports_dir / "lotofacil_optimizer.xlsx"

    @property
    def prediction_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_prediction.csv"

    @property
    def prediction_report_path(self) -> Path:
        return self.exports_dir / "lotofacil_prediction_report.md"

    @property
    def prediction_excel_path(self) -> Path:
        return self.exports_dir / "lotofacil_prediction.xlsx"

    @property
    def single_prediction_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_prediction_single.csv"

    @property
    def single_prediction_report_path(self) -> Path:
        return self.exports_dir / "lotofacil_prediction_single_report.md"

    @property
    def single_prediction_excel_path(self) -> Path:
        return self.exports_dir / "lotofacil_prediction_single.xlsx"

    @property
    def mandel_plan_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_mandel_plan.csv"

    @property
    def mandel_games_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_mandel_games.csv"

    @property
    def mandel_report_path(self) -> Path:
        return self.exports_dir / "lotofacil_mandel_report.md"

    @property
    def mandel_excel_path(self) -> Path:
        return self.exports_dir / "lotofacil_mandel.xlsx"

    @property
    def generated_games_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_jogos_gerados.csv"

    @property
    def generated_games_excel_path(self) -> Path:
        return self.exports_dir / "lotofacil_jogos_gerados.xlsx"

    @property
    def full_export_excel_path(self) -> Path:
        return self.exports_dir / "lotofacil_analytics_completo.xlsx"

    @property
    def post_result_games_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_pos_sorteio_jogos.csv"

    @property
    def post_result_dezenas_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_pos_sorteio_dezenas.csv"

    @property
    def post_result_report_path(self) -> Path:
        return self.exports_dir / "lotofacil_pos_sorteio_report.md"

    @property
    def post_result_excel_path(self) -> Path:
        return self.exports_dir / "lotofacil_pos_sorteio.xlsx"

    @property
    def final_backtest_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_backtest_final_score.csv"

    @property
    def final_backtest_summary_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_backtest_final_score_summary.csv"

    @property
    def final_backtest_excel_path(self) -> Path:
        return self.exports_dir / "lotofacil_backtest_final_score.xlsx"

    @property
    def exhaustive_backtest_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_backtest_exaustivo_single.csv"

    @property
    def exhaustive_backtest_summary_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_backtest_exaustivo_single_summary.csv"

    @property
    def exhaustive_backtest_excel_path(self) -> Path:
        return self.exports_dir / "lotofacil_backtest_exaustivo_single.xlsx"

    @property
    def ablation_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_ablation_test.csv"

    @property
    def ablation_summary_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_ablation_test_summary.csv"

    @property
    def ablation_excel_path(self) -> Path:
        return self.exports_dir / "lotofacil_ablation_test.xlsx"

    @property
    def tune_weights_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_tune_weights_results.csv"

    @property
    def tune_weights_summary_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_tune_weights_summary.csv"

    @property
    def tune_weights_excel_path(self) -> Path:
        return self.exports_dir / "lotofacil_tune_weights.xlsx"

    @property
    def tuned_weights_json_path(self) -> Path:
        return self.processed_dir / "lotofacil_tuned_weights.json"

    @property
    def supervised_calibration_state_json_path(self) -> Path:
        return self.processed_dir / "lotofacil_supervised_calibration_state.json"

    @property
    def supervised_calibration_results_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_supervised_calibration_results.csv"

    @property
    def supervised_calibration_summary_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_supervised_calibration_summary.csv"

    @property
    def supervised_calibration_weights_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_supervised_calibration_weights.csv"

    @property
    def supervised_calibration_weights_json_path(self) -> Path:
        return self.processed_dir / "lotofacil_supervised_calibrated_weights.json"

    @property
    def supervised_calibration_excel_path(self) -> Path:
        return self.exports_dir / "lotofacil_supervised_calibration.xlsx"

    @property
    def top100_prediction_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_prediction_top100.csv"

    @property
    def top100_prediction_report_path(self) -> Path:
        return self.exports_dir / "lotofacil_prediction_top100_report.md"

    @property
    def top100_prediction_excel_path(self) -> Path:
        return self.exports_dir / "lotofacil_prediction_top100.xlsx"

    @property
    def top100_prediction_history_dir(self) -> Path:
        return self.processed_dir / "top100_history"

    @property
    def top100_backtest_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_backtest_top100.csv"

    @property
    def top100_backtest_summary_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_backtest_top100_summary.csv"

    @property
    def top100_backtest_excel_path(self) -> Path:
        return self.exports_dir / "lotofacil_backtest_top100.xlsx"

    @property
    def top50_refinement_state_json_path(self) -> Path:
        return self.processed_dir / "lotofacil_top50_refinement_state.json"

    @property
    def top50_refinement_results_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_top50_refinement_results.csv"

    @property
    def top50_refinement_summary_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_top50_refinement_summary.csv"

    @property
    def top50_refinement_weights_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_top50_refinement_weights.csv"

    @property
    def top50_refinement_weights_json_path(self) -> Path:
        return self.processed_dir / "lotofacil_top50_refinement_weights.json"

    @property
    def top50_refinement_excel_path(self) -> Path:
        return self.exports_dir / "lotofacil_top50_refinement.xlsx"

    @property
    def top100_repair_state_json_path(self) -> Path:
        return self.processed_dir / "lotofacil_top100_repair_state.json"

    @property
    def top100_repair_weights_json_path(self) -> Path:
        return self.processed_dir / "lotofacil_top100_repair_weights.json"

    @property
    def top100_repair_results_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_top100_repair_results.csv"

    @property
    def top100_repair_summary_csv_path(self) -> Path:
        return self.processed_dir / "lotofacil_top100_repair_summary.csv"
