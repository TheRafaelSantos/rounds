from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import logging

import pandas as pd

from lotofacil_analytics.auditoria import build_auditoria
from lotofacil_analytics.backtest_lotofacil import compute_hits, run_backtest
from lotofacil_analytics.calibration_lab import (
    WEIGHT_COMPONENTS,
    _append_csv,
    _apply_calibration_novelty,
    _read_csv,
    _sync_elites_from_attempts,
    _weights_for_attempt,
    load_calibration_lab_status,
    run_calibration_lab,
)
from lotofacil_analytics.calibration_pilot import run_calibration_pilot
from lotofacil_analytics.climate_pipeline import ClimatePipeline
from lotofacil_analytics.combinacoes import build_combinacoes_features, build_combinacoes_outputs
from lotofacil_analytics.config import AppConfig
from lotofacil_analytics.context_features import build_context_model, score_contextual_candidate
from lotofacil_analytics.dezenas_history import build_dezenas_historico, build_dezenas_long
from lotofacil_analytics.decision_layer import (
    build_single_prediction,
    run_ablation_test,
    run_exhaustive_single_backtest,
    run_weight_tuning,
    weights_for_profile,
)
from lotofacil_analytics.engine_calibration import run_engine_calibration
from lotofacil_analytics.exhaustive_optimizer import EXHAUSTIVE_SOURCE_MODEL, build_exhaustive_candidates, resolve_exhaustive_weights
from lotofacil_analytics.features_base import build_base_features
from lotofacil_analytics.games import generate_games
from lotofacil_analytics.interface_web import _html_page
from lotofacil_analytics.final_backtest import run_final_score_backtest
from lotofacil_analytics.mandel_strategy import build_plan_table, choose_strategy_universe, greedy_reduced_closure, run_mandel_strategy
from lotofacil_analytics.ml_temporal import run_ml_temporal
from lotofacil_analytics.optimizer import build_optimized_candidates, score_candidate
from lotofacil_analytics.post_result_analysis import analyze_post_result, parse_numbers
from lotofacil_analytics.predictor import select_final_games
from lotofacil_analytics.selection_guard import build_number_guard_table, enrich_candidates_with_decision_guard
from lotofacil_analytics.temporal_deep import build_temporal_deep_rows, temporal_deep_number_scores
from lotofacil_analytics.normalize import normalize_contest
from lotofacil_analytics.transition_analysis import build_transition_model, build_transition_outputs, score_transition_candidate
from lotofacil_analytics.validators import DataValidationError, validate_contest_record, validate_dataset


def sample_payload(numero: int = 1) -> dict:
    return {
        "numero": numero,
        "dataApuracao": "29/09/2003",
        "dataProximoConcurso": "06/10/2003",
        "tipoJogo": "LOTOFACIL",
        "acumulado": False,
        "localSorteio": "Caminhao da Sorte",
        "nomeMunicipioUFSorteio": "CRUZ ALTA, RS",
        "indicadorConcursoEspecial": 1,
        "listaDezenas": ["02", "03", "05", "06", "09", "10", "11", "13", "14", "16", "18", "20", "23", "24", "25"],
        "dezenasSorteadasOrdemSorteio": ["18", "20", "25", "23", "10", "11", "24", "14", "06", "02", "13", "09", "05", "16", "03"],
        "listaRateioPremio": [
            {"faixa": 1, "numeroDeGanhadores": 5, "valorPremio": 49765.82},
            {"faixa": 2, "numeroDeGanhadores": 154, "valorPremio": 689.84},
            {"faixa": 3, "numeroDeGanhadores": 4645, "valorPremio": 10.0},
            {"faixa": 4, "numeroDeGanhadores": 48807, "valorPremio": 4.0},
            {"faixa": 5, "numeroDeGanhadores": 257593, "valorPremio": 2.0},
        ],
    }


def payload_with_dezenas(numero: int, dezenas: list[str]) -> dict:
    payload = sample_payload(numero)
    payload["listaDezenas"] = list(dezenas)
    payload["dezenasSorteadasOrdemSorteio"] = list(dezenas)
    return payload


def cyclic_dezenas(offset: int) -> list[str]:
    nums = [((offset + idx - 1) % 25) + 1 for idx in range(15)]
    return [f"{n:02d}" for n in sorted(nums)]


class LotofacilValidationTest(unittest.TestCase):
    def test_normalize_and_validate_valid_payload(self) -> None:
        record = normalize_contest(sample_payload())
        validate_contest_record(record)
        self.assertEqual(record["concurso"], 1)
        self.assertEqual(record["d01"], 2)
        self.assertEqual(record["d15"], 25)
        self.assertEqual(record["ganhadores_15"], 5)

    def test_reject_duplicate_dezena(self) -> None:
        payload = sample_payload()
        payload["listaDezenas"] = ["02"] * 15
        with self.assertRaises(ValueError):
            normalize_contest(payload)

    def test_reject_duplicate_concurso(self) -> None:
        record = normalize_contest(sample_payload())
        with self.assertRaises(DataValidationError):
            validate_dataset([record, record])

    def test_build_base_features_uses_previous_contest_only(self) -> None:
        first = normalize_contest(sample_payload(1))
        second_payload = sample_payload(2)
        second_payload["listaDezenas"] = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15"]
        second_payload["dezenasSorteadasOrdemSorteio"] = list(second_payload["listaDezenas"])
        second = normalize_contest(second_payload)

        features = build_base_features(pd.DataFrame([first, second]))

        self.assertEqual(int(features.loc[0, "qtd_repetidas_anterior"]), 0)
        self.assertTrue(pd.isna(features.loc[0, "qtd_novas_vs_anterior"]))
        self.assertEqual(int(features.loc[1, "qtd_repetidas_anterior"]), 9)
        self.assertEqual(int(features.loc[1, "qtd_pares"]), 7)
        self.assertEqual(int(features.loc[1, "faixa_01_05"]), 5)
        self.assertEqual(int(features.loc[1, "maior_sequencia_consecutiva"]), 15)

    def test_build_dezenas_history_uses_only_previous_draws(self) -> None:
        first = normalize_contest(sample_payload(1))
        second_payload = sample_payload(2)
        second_payload["listaDezenas"] = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15"]
        second_payload["dezenasSorteadasOrdemSorteio"] = list(second_payload["listaDezenas"])
        second = normalize_contest(second_payload)
        concursos = pd.DataFrame([first, second])

        dezenas_long = build_dezenas_long(concursos)
        historico = build_dezenas_historico(concursos)

        self.assertEqual(len(dezenas_long), 30)
        self.assertEqual(len(historico), 50)

        concurso_1_dezena_2 = historico[(historico["concurso"] == 1) & (historico["dezena"] == 2)].iloc[0]
        self.assertEqual(int(concurso_1_dezena_2["saiu_no_concurso"]), 1)
        self.assertEqual(int(concurso_1_dezena_2["freq_total_ate_anterior"]), 0)
        self.assertEqual(int(concurso_1_dezena_2["nunca_saiu_ate_anterior"]), 1)

        concurso_2_dezena_2 = historico[(historico["concurso"] == 2) & (historico["dezena"] == 2)].iloc[0]
        self.assertEqual(int(concurso_2_dezena_2["saiu_no_concurso"]), 1)
        self.assertEqual(int(concurso_2_dezena_2["freq_total_ate_anterior"]), 1)
        self.assertEqual(int(concurso_2_dezena_2["freq_ultimos_5"]), 1)
        self.assertEqual(int(concurso_2_dezena_2["saiu_concurso_anterior"]), 1)
        self.assertEqual(int(concurso_2_dezena_2["atraso_atual"]), 0)

        concurso_2_dezena_1 = historico[(historico["concurso"] == 2) & (historico["dezena"] == 1)].iloc[0]
        self.assertEqual(int(concurso_2_dezena_1["freq_total_ate_anterior"]), 0)
        self.assertEqual(int(concurso_2_dezena_1["nunca_saiu_ate_anterior"]), 1)
        self.assertEqual(int(concurso_2_dezena_1["atraso_atual"]), 1)

    def test_build_combinacoes_features_uses_previous_draws(self) -> None:
        first = normalize_contest(sample_payload(1))
        second_payload = sample_payload(2)
        second_payload["listaDezenas"] = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15"]
        second_payload["dezenasSorteadasOrdemSorteio"] = list(second_payload["listaDezenas"])
        second = normalize_contest(second_payload)
        concursos = pd.DataFrame([first, second])

        features = build_combinacoes_features(concursos)

        self.assertEqual(int(features.loc[0, "qtd_pares_combinatorios"]), 105)
        self.assertEqual(int(features.loc[0, "qtd_pares_ineditos_ate_entao"]), 105)
        self.assertEqual(int(features.loc[0, "maior_freq_par_ate_anterior"]), 0)
        self.assertEqual(int(features.loc[1, "qtd_pares_ineditos_ate_entao"]), 69)
        self.assertEqual(int(features.loc[1, "qtd_trios_ineditos_ate_entao"]), 371)
        self.assertEqual(int(features.loc[1, "qtd_quartetos_ineditos_ate_entao"]), 1239)

    def test_build_combinacoes_aggregates_all_possible_combos(self) -> None:
        first = normalize_contest(sample_payload(1))
        second_payload = sample_payload(2)
        second_payload["listaDezenas"] = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15"]
        second_payload["dezenasSorteadasOrdemSorteio"] = list(second_payload["listaDezenas"])
        second = normalize_contest(second_payload)

        _, pares, trios, quartetos = build_combinacoes_outputs(pd.DataFrame([first, second]))

        self.assertEqual(len(pares), 300)
        self.assertEqual(len(trios), 2300)
        self.assertEqual(len(quartetos), 12650)
        pair_02_03 = pares[pares["combo"] == "02-03"].iloc[0]
        pair_01_02 = pares[pares["combo"] == "01-02"].iloc[0]
        self.assertEqual(int(pair_02_03["freq_total_historico"]), 2)
        self.assertEqual(int(pair_01_02["freq_total_historico"]), 1)

    def test_compute_hits(self) -> None:
        self.assertEqual(compute_hits([1, 2, 3], [3, 4, 5]), 1)
        self.assertEqual(compute_hits([1, 2, 3], [4, 5, 6]), 0)

    def test_parse_numbers_accepts_common_separators(self) -> None:
        nums = parse_numbers("01 - 02 - 03, 04 05 06 07 08 09 10 11 12 13 14 15")

        self.assertEqual(nums, list(range(1, 16)))

    def test_analyze_post_result_writes_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            predictions_path = base / "prediction.csv"
            predictions = pd.DataFrame(
                [
                    {"jogo": 1, "nums": "01 02 03 04 05 06 07 08 09 10 11 12 13 14 15", "score_final": 99.0},
                    {"jogo": 2, "nums": "11 12 13 14 15 16 17 18 19 20 21 22 23 24 25", "score_final": 90.0},
                ]
            )
            predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")

            summary = analyze_post_result(
                actual_numbers="01 - 02 - 03 - 04 - 05 - 06 - 07 - 08 - 09 - 10 - 11 - 12 - 13 - 14 - 15",
                predictions_path=predictions_path,
                optimizer_candidates_path=base / "missing_optimizer.csv",
                games_csv_path=base / "jogos.csv",
                dezenas_csv_path=base / "dezenas.csv",
                report_path=base / "report.md",
                excel_path=base / "analise.xlsx",
                label="teste",
                concurso=999,
            )

            self.assertEqual(summary.best_hits, 15)
            self.assertEqual(summary.union_hits, 15)
            self.assertTrue((base / "jogos.csv").exists())
            self.assertTrue((base / "dezenas.csv").exists())
            self.assertTrue((base / "report.md").exists())
            self.assertTrue((base / "analise.xlsx").exists())

    def test_run_backtest_returns_all_methods(self) -> None:
        rows = [
            normalize_contest(payload_with_dezenas(1, ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15"])),
            normalize_contest(payload_with_dezenas(2, ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16"])),
            normalize_contest(payload_with_dezenas(3, ["03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17"])),
            normalize_contest(payload_with_dezenas(4, ["04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18"])),
        ]
        results, summary = run_backtest(pd.DataFrame(rows), n_eval=2, min_history=2, seed=123, window=2, candidates=100)

        self.assertEqual(int(results["concurso_previsto"].nunique()), 2)
        self.assertEqual(len(results), 10)
        self.assertEqual(set(results["modelo_nome"]), {"aleatorio_puro", "frequencia_quente", "frequencia_fria", "hibrido_quente_frio", "balanceado_basico"})
        self.assertEqual(set(summary["modelo_nome"]), set(results["modelo_nome"]))
        self.assertTrue((results["qtd_acertos"] >= 0).all())

    def test_build_auditoria_outputs_expected_tables(self) -> None:
        rows = [
            normalize_contest(payload_with_dezenas(1, ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15"])),
            normalize_contest(payload_with_dezenas(2, ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16"])),
            normalize_contest(payload_with_dezenas(3, ["03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17"])),
        ]

        resumo, dezenas, anomalias, monte_carlo = build_auditoria(pd.DataFrame(rows), monte_carlo_runs=5, seed=123)

        self.assertEqual(len(dezenas), 25)
        self.assertEqual(len(monte_carlo), 5)
        self.assertIn("chi_square_p_value_aprox", set(resumo["metrica"]))
        self.assertIn("entropia_ratio", set(resumo["metrica"]))
        self.assertTrue((dezenas["freq_observada"] >= 0).all())
        self.assertIsNotNone(anomalias)

    def test_run_ml_temporal_outputs_predictions(self) -> None:
        rows = []
        for idx in range(1, 8):
            start = idx
            dezenas = [f"{n:02d}" for n in range(start, start + 15)]
            rows.append(normalize_contest(payload_with_dezenas(idx, dezenas)))

        dataset, predictions, summary = run_ml_temporal(
            pd.DataFrame(rows),
            train_ratio=0.50,
            validation_ratio=0.25,
            epochs=5,
            learning_rate=0.01,
            l2=0.001,
            seed=123,
        )

        self.assertEqual(len(dataset), 175)
        self.assertFalse(predictions.empty)
        self.assertFalse(summary.empty)
        self.assertIn("ml_logistico_simples", set(predictions["modelo_nome"]))
        self.assertTrue((predictions["qtd_acertos"] >= 0).all())

    def test_optimizer_generates_valid_candidates(self) -> None:
        rows = []
        for idx in range(1, 8):
            start = idx
            dezenas = [f"{n:02d}" for n in range(start, start + 15)]
            rows.append(normalize_contest(payload_with_dezenas(idx, dezenas)))

        candidates, summary = build_optimized_candidates(
            pd.DataFrame(rows),
            seed=123,
            candidate_pool=200,
            top_games=10,
            generations=2,
            population=10,
        )

        self.assertEqual(len(candidates), 10)
        self.assertFalse(summary.empty)
        for nums_text in candidates["nums"].tolist():
            nums = [int(part) for part in nums_text.split()]
            self.assertEqual(len(nums), 15)
            self.assertEqual(len(set(nums)), 15)
            self.assertTrue(all(1 <= n <= 25 for n in nums))

    def test_exhaustive_optimizer_uses_context_and_limited_scan(self) -> None:
        rows = []
        for idx in range(1, 14):
            payload = payload_with_dezenas(idx, cyclic_dezenas(idx))
            payload["nomeMunicipioUFSorteio"] = "SAO PAULO, SP"
            payload["localSorteio"] = "ESPAÇO DA SORTE"
            rows.append(normalize_contest(payload))

        candidates, summary = build_exhaustive_candidates(
            pd.DataFrame(rows),
            top_games=5,
            limit_combinations=200,
            weights={"localidade_numerologia": 0.30, "estatistico": 0.10},
        )

        self.assertEqual(len(candidates), 5)
        self.assertEqual(set(candidates["source_model"]), {EXHAUSTIVE_SOURCE_MODEL})
        self.assertIn("score_localidade_numerologia", candidates.columns)
        self.assertIn("score_climatico", candidates.columns)
        self.assertIn("score_temporal_profundo", candidates.columns)
        self.assertIn("score_transicao", candidates.columns)
        self.assertIn("contexto_cidade_sorteio", candidates.columns)
        self.assertIn("score_weights", set(summary["metrica"]))
        self.assertIn("transicao_media_repetidas", set(summary["metrica"]))
        self.assertEqual(int(summary[summary["metrica"] == "combinacoes_avaliadas"]["valor"].iloc[0]), 200)
        for nums_text in candidates["nums"].tolist():
            nums = [int(part) for part in nums_text.split()]
            self.assertEqual(len(nums), 15)
            self.assertEqual(len(set(nums)), 15)
            self.assertTrue(all(1 <= n <= 25 for n in nums))

    def test_context_model_uses_climate_features_without_network(self) -> None:
        rows = []
        climate_rows = []
        for idx in range(1, 14):
            payload = payload_with_dezenas(idx, cyclic_dezenas(idx))
            payload["nomeMunicipioUFSorteio"] = "SAO PAULO, SP"
            rows.append(normalize_contest(payload))
            climate_rows.append(
                {
                    "concurso": idx,
                    "clima_status": "ok",
                    "clima_temperature_2m": 24.0,
                    "clima_apparent_temperature": 25.0,
                    "clima_relative_humidity_2m": 70.0,
                    "clima_surface_pressure": 930.0,
                    "clima_precipitation": 0.0,
                    "clima_temperature_media_30d": 22.0,
                    "clima_temperature_anomalia": 2.0,
                    "clima_temperatura_faixa": "quente",
                    "clima_sensacao_faixa": "quente",
                    "clima_umidade_faixa": "umido",
                    "clima_pressao_faixa": "pressao_normal",
                    "clima_chuva_faixa": "sem_chuva",
                    "clima_anomalia_faixa": "normal",
                    "clima_assinatura": "quente|quente|umido|pressao_normal|sem_chuva|normal",
                }
            )
        target_climate = dict(climate_rows[-1])
        target_climate["concurso"] = 14

        model = build_context_model(
            pd.DataFrame(rows),
            climate_features=pd.DataFrame(climate_rows),
            target_climate=target_climate,
        )
        detail = score_contextual_candidate(list(range(1, 16)), model)

        self.assertEqual(model.target.clima_temperatura_faixa, "quente")
        self.assertIn("score_climatico", detail)
        self.assertIn("contexto_clima_temperature_2m", detail)
        self.assertGreaterEqual(float(detail["score_climatico"]), 0.0)
        self.assertLessEqual(float(detail["score_climatico"]), 100.0)

    def test_climate_pipeline_is_incremental_by_default(self) -> None:
        first = normalize_contest(payload_with_dezenas(1, cyclic_dezenas(1)))
        second = normalize_contest(payload_with_dezenas(2, cyclic_dezenas(2)))
        concursos = pd.DataFrame([first, second])
        existing = pd.DataFrame(
            [
                {
                    "concurso": 1,
                    "data_sorteio": first["data_sorteio"],
                    "cidade_sorteio": "SAO PAULO",
                    "uf_sorteio": "SP",
                    "clima_status": "ok",
                    "clima_temperature_2m": 22.0,
                    "clima_temperatura_faixa": "quente",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig.from_base_dir(Path(tmp))
            config.processed_csv_path.parent.mkdir(parents=True, exist_ok=True)
            concursos.to_csv(config.processed_csv_path, index=False, encoding="utf-8-sig")
            existing.to_csv(config.climate_csv_path, index=False, encoding="utf-8-sig")

            def fake_build_climate_features(concursos_to_process: pd.DataFrame, **kwargs):
                self.assertEqual(concursos_to_process["concurso"].astype(int).tolist(), [2])
                new = existing.copy()
                new["concurso"] = 2
                new["data_sorteio"] = second["data_sorteio"]
                new["clima_temperature_2m"] = 23.0
                summary = pd.DataFrame(
                    [
                        {
                            "cidade": "SAO PAULO",
                            "uf": "SP",
                            "concursos": 1,
                            "localidades_total": 1,
                            "localidades_processadas_planejadas": 1,
                            "status": "ok",
                            "erro": "",
                            "linhas_clima": 1,
                        }
                    ]
                )
                return new, summary

            with patch("lotofacil_analytics.climate_pipeline.build_climate_features", side_effect=fake_build_climate_features):
                summary = ClimatePipeline(config=config, logger=logging.getLogger("test")).run()

            saved = pd.read_csv(config.climate_csv_path, encoding="utf-8-sig")
            self.assertEqual(saved["concurso"].astype(int).tolist(), [1, 2])
        self.assertEqual(summary.rows, 2)

    def test_temporal_deep_outputs_dynamic_history(self) -> None:
        rows = []
        for idx in range(1, 8):
            rows.append(normalize_contest(payload_with_dezenas(idx, cyclic_dezenas(idx))))
        concursos = pd.DataFrame(rows)

        deep = build_temporal_deep_rows(concursos)
        scores = temporal_deep_number_scores(concursos.iloc[:-1].copy(), target_date=concursos.iloc[-1]["data_sorteio"])

        self.assertEqual(len(deep), 7 * 25)
        self.assertIn("freq_mesmo_dia_semana_ate_anterior", deep.columns)
        self.assertIn("freq_ultimos_15_dias_ate_anterior", deep.columns)
        self.assertIn("freq_mesmo_bimestre_ate_anterior", deep.columns)
        self.assertEqual(set(scores), set(range(1, 26)))
        self.assertTrue(all(0.0 <= float(value) <= 100.0 for value in scores.values()))

    def test_engine_calibration_generates_normalized_weights(self) -> None:
        rows = []
        for idx in range(1, 16):
            rows.append(normalize_contest(payload_with_dezenas(idx, cyclic_dezenas(idx))))
        concursos = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmp:
            weights_path = Path(tmp) / "weights.json"
            results, summary, payload = run_engine_calibration(
                concursos,
                climate_features=pd.DataFrame(),
                from_concurso=12,
                to_concurso=14,
                baseline_samples=2,
                seed=123,
                weights_json_path=weights_path,
            )

            self.assertFalse(results.empty)
            self.assertFalse(summary.empty)
            self.assertTrue(weights_path.exists())
            self.assertIn("weights", payload)
            self.assertAlmostEqual(sum(payload["weights"].values()), 1.0, places=6)

    def test_calibration_pilot_resumes_from_saved_attempts(self) -> None:
        rows = []
        for idx in range(1, 18):
            rows.append(normalize_contest(payload_with_dezenas(idx, cyclic_dezenas(idx))))
        concursos = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            paths = {
                "candidates_csv_path": base / "pilot_candidates.csv",
                "results_csv_path": base / "pilot_results.csv",
                "summary_csv_path": base / "pilot_summary.csv",
                "state_json_path": base / "pilot_state.json",
                "excel_path": base / "pilot.xlsx",
            }
            first = run_calibration_pilot(
                concursos,
                climate_features=pd.DataFrame(),
                target_concurso=14,
                attempts=2,
                candidate_pool=12,
                exhaustive_limit=200,
                max_overlap=14,
                seed=123,
                draw_hour=20,
                draw_minute=0,
                reset=True,
                **paths,
            )
            second = run_calibration_pilot(
                concursos,
                climate_features=pd.DataFrame(),
                target_concurso=14,
                attempts=4,
                candidate_pool=12,
                exhaustive_limit=200,
                max_overlap=14,
                seed=123,
                draw_hour=20,
                draw_minute=0,
                reset=False,
                **paths,
            )

            results = pd.read_csv(paths["results_csv_path"])
            self.assertEqual(first.completed_attempts, 2)
            self.assertEqual(first.attempts_this_run, 2)
            self.assertEqual(second.completed_attempts, 4)
            self.assertEqual(second.attempts_this_run, 2)
            self.assertEqual(results["tentativa"].astype(int).tolist(), [1, 2, 3, 4])
            self.assertTrue(paths["state_json_path"].exists())
            self.assertTrue(paths["excel_path"].exists())

    def test_calibration_lab_writes_resumable_status(self) -> None:
        rows = []
        for idx in range(1, 18):
            rows.append(normalize_contest(payload_with_dezenas(idx, cyclic_dezenas(idx))))
        concursos = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            paths = {
                "state_json_path": base / "lab_state.json",
                "attempts_csv_path": base / "lab_attempts.csv",
                "winners_csv_path": base / "lab_winners.csv",
                "elites_csv_path": base / "lab_elites.csv",
                "summary_csv_path": base / "lab_summary.csv",
                "average_weights_csv_path": base / "lab_average_weights.csv",
                "excel_path": base / "lab.xlsx",
                "engine_weights_json_path": base / "engine_weights.json",
                "cache_dir": base / "lab_cache",
            }
            summary = run_calibration_lab(
                concursos,
                climate_features=pd.DataFrame(),
                from_concurso=14,
                to_concurso=14,
                max_attempts=2,
                top_games=12,
                exhaustive_limit=220,
                max_overlap=14,
                seed=123,
                draw_hour=20,
                draw_minute=0,
                min_history=10,
                max_runtime_seconds=0,
                reset=True,
                **paths,
            )
            status = load_calibration_lab_status(
                state_json_path=paths["state_json_path"],
                attempts_csv_path=paths["attempts_csv_path"],
                winners_csv_path=paths["winners_csv_path"],
                elites_csv_path=paths["elites_csv_path"],
                average_weights_csv_path=paths["average_weights_csv_path"],
                engine_weights_json_path=paths["engine_weights_json_path"],
            )

            attempts = pd.read_csv(paths["attempts_csv_path"])
            self.assertEqual(summary.attempts_this_run, 2)
            self.assertEqual(len(attempts), 2)
            self.assertEqual(attempts.loc[0, "cache_status"], "built")
            self.assertEqual(attempts.loc[1, "cache_status"], "hit")
            self.assertIn("status", status["state"])
            self.assertEqual(len(status["recent_attempts"]), 2)
            self.assertTrue(paths["state_json_path"].exists())
            self.assertTrue(paths["summary_csv_path"].exists())
            self.assertTrue(paths["average_weights_csv_path"].exists())
            self.assertTrue((paths["cache_dir"] / "concurso_14" / "scores.npy").exists())

    def test_calibration_lab_saves_elites_from_11_to_15_hits(self) -> None:
        weights = resolve_exhaustive_weights({component: 1.0 for component in WEIGHT_COMPONENTS})
        rows = []
        for attempt, hits in enumerate([10, 11, 12, 13, 14, 15], start=1):
            row = {
                "target_concurso": 2500,
                "tentativa": attempt,
                "melhor_acerto": hits,
                "melhor_jogo": "01 02 03 04 05 06 07 08 09 10 11 12 13 14 15",
                "jogo_1": "01 02 03 04 05 06 07 08 09 10 11 12 13 14 15",
                "acertos_jogo_1": hits,
                "jogo_2": "01 02 03 04 05 06 07 08 09 10 11 12 13 14 16",
                "acertos_jogo_2": max(0, hits - 1),
                "weight_strategy": "test",
                "score_weights": "test",
            }
            for component, value in weights.items():
                row[f"peso_{component}"] = value
            rows.append(row)

        with tempfile.TemporaryDirectory() as tmp:
            elites = _sync_elites_from_attempts(
                attempts=pd.DataFrame(rows),
                elites_csv_path=Path(tmp) / "lab_elites.csv",
            )

            self.assertEqual(elites["melhor_acerto"].astype(int).tolist(), [15, 14, 13, 12, 11])
            self.assertNotIn(10, elites["melhor_acerto"].astype(int).tolist())

    def test_calibration_lab_uses_elite_weights_as_anchor(self) -> None:
        weights = resolve_exhaustive_weights({component: index + 1 for index, component in enumerate(WEIGHT_COMPONENTS)})
        row = {
            "target_concurso": 2500,
            "tentativa": 7,
            "melhor_acerto": 12,
            "melhor_jogo": "01 02 03 04 05 06 07 08 09 10 11 12 13 14 15",
            "weight_strategy": "test",
        }
        for component, value in weights.items():
            row[f"peso_{component}"] = value

        with tempfile.TemporaryDirectory() as tmp:
            elites = _sync_elites_from_attempts(
                attempts=pd.DataFrame([row]),
                elites_csv_path=Path(tmp) / "lab_elites.csv",
            )
            selected_weights = {}
            meta = {}
            for seed in range(50):
                selected_weights, meta = _weights_for_attempt(
                    target_concurso=2500,
                    attempt=20,
                    seed=seed,
                    average_winner_weights=None,
                    best_current_weights=None,
                    elite_rows=elites,
                )
                if str(meta.get("weight_strategy", "")).startswith("elite_"):
                    break

            self.assertTrue(str(meta.get("weight_strategy", "")).startswith("elite_"))
            self.assertEqual(str(meta.get("elite_source_hits", "")), "12")
            self.assertAlmostEqual(sum(selected_weights.values()), 1.0, places=6)

    def test_calibration_lab_penalizes_repeated_games(self) -> None:
        repeated = "01 02 03 04 05 06 07 08 09 10 11 12 13 14 15"
        fresh = "01 02 03 04 05 06 07 08 09 16 17 18 19 20 21"
        alternate = "01 02 03 04 05 06 07 08 10 16 17 18 19 20 22"
        attempts = pd.DataFrame(
            [
                {"target_concurso": 2500, "jogo_1": repeated, "jogo_2": alternate},
                {"target_concurso": 2500, "jogo_1": repeated, "jogo_2": alternate},
                {"target_concurso": 2500, "jogo_1": repeated, "jogo_2": alternate},
            ]
        )
        candidates = pd.DataFrame(
            [
                {"nums": repeated, "score_final": 99.0, "score_transicao": 99.0, "score_contextual": 99.0},
                {"nums": fresh, "score_final": 97.0, "score_transicao": 97.0, "score_contextual": 97.0},
                {"nums": alternate, "score_final": 96.0, "score_transicao": 96.0, "score_contextual": 96.0},
            ]
        )

        penalized = _apply_calibration_novelty(candidates, attempts=attempts, target_concurso=2500)
        repeated_row = penalized.loc[penalized["nums"] == repeated].iloc[0]
        fresh_row = penalized.loc[penalized["nums"] == fresh].iloc[0]
        final_games = select_final_games(penalized, max_overlap=10)

        self.assertGreater(int(repeated_row["calibration_repeat_count"]), 0)
        self.assertLess(float(repeated_row["score_final"]), float(fresh_row["score_final"]))
        self.assertNotEqual(final_games.loc[0, "nums"], repeated)

    def test_calibration_lab_attempt_csv_evolves_cache_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempts.csv"
            old_row = {
                "target_concurso": 2500,
                "tentativa": 1,
                "score_jogo_1": 90.0,
                "score_jogo_2": 88.0,
                "score_weights": "estatistico=1.0000",
                "peso_estatistico": 1.0,
            }
            new_row = {
                **old_row,
                "tentativa": 2,
                "cache_status": "hit",
                "cache_rows": 100,
                "cache_path": "cache/concurso_2500",
            }

            _append_csv(path, old_row)
            _append_csv(path, new_row)
            migrated = _read_csv(path)

            self.assertEqual(len(migrated), 2)
            self.assertIn("cache_status", migrated.columns)
            self.assertEqual(str(migrated.loc[1, "cache_status"]), "hit")
            self.assertEqual(str(migrated.loc[1, "score_weights"]), "estatistico=1.0000")

    def test_calibration_lab_append_preserves_existing_column_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempts.csv"
            first = {
                "target_concurso": 2500,
                "score_weights": "estatistico=1.0000",
                "peso_estatistico": 1.0,
                "weight_strategy": "preset",
            }
            second = {
                "target_concurso": 2500,
                "weight_strategy": "elite_mutation",
                "score_weights": "estatistico=0.5000",
                "peso_estatistico": 0.5,
            }

            _append_csv(path, first)
            _append_csv(path, second)
            rows = path.read_text(encoding="utf-8-sig").splitlines()

            self.assertEqual(rows[0].split(","), ["target_concurso", "score_weights", "peso_estatistico", "weight_strategy"])
            self.assertEqual(rows[2].split(","), ["2500", "estatistico=0.5000", "0.5", "elite_mutation"])

    def test_calibration_lab_read_csv_repairs_misaligned_weight_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempts.csv"
            header = [
                "score_weights",
                "peso_estatistico",
                "peso_historico",
                "peso_atraso",
                "peso_combinatorio",
                "peso_localidade_numerologia",
                "peso_climatico",
                "peso_temporal_profundo",
                "peso_cenarios",
                "peso_contrarian",
                "peso_transicao",
                "peso_nao_repeticao_exata",
                "weight_strategy",
                "elite_source_attempts",
                "elite_source_hits",
            ]
            misaligned = [
                "elite_crossover",
                "101|202",
                "12|11",
                "estatistico=0.1;historico=0.2",
                "0.1",
                "0.2",
                "0.3",
                "0.4",
                "0.5",
                "0.6",
                "0.7",
                "0.8",
                "0.9",
                "0.10",
                "0.11",
            ]
            path.write_text(",".join(header) + "\n" + ",".join(misaligned) + "\n", encoding="utf-8")

            repaired = _read_csv(path)

            self.assertEqual(str(repaired.loc[0, "weight_strategy"]), "elite_crossover")
            self.assertEqual(str(repaired.loc[0, "elite_source_attempts"]), "101|202")
            self.assertEqual(str(repaired.loc[0, "elite_source_hits"]), "12|11")
            self.assertEqual(str(repaired.loc[0, "score_weights"]), "estatistico=0.1;historico=0.2")
            self.assertEqual(float(repaired.loc[0, "peso_estatistico"]), 0.1)
            self.assertEqual(float(repaired.loc[0, "peso_nao_repeticao_exata"]), 0.11)

    def test_calibration_lab_read_csv_tolerates_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.csv"
            path.write_text("", encoding="utf-8")

            df = _read_csv(path)

            self.assertTrue(df.empty)

    def test_transition_outputs_compare_consecutive_draws(self) -> None:
        rows = []
        for idx in range(1, 6):
            rows.append(normalize_contest(payload_with_dezenas(idx, cyclic_dezenas(idx))))
        concursos = pd.DataFrame(rows)

        transitions, summary, number_stats = build_transition_outputs(concursos)
        model = build_transition_model(concursos)
        score = score_transition_candidate(cyclic_dezenas(6), model)

        self.assertEqual(len(transitions), 4)
        self.assertEqual(len(number_stats), 25)
        self.assertIn("qtd_repetidas", transitions.columns)
        self.assertIn("probabilidade_ficar_suavizada", number_stats.columns)
        self.assertIn("media_repetidas", set(summary["metrica"]))
        self.assertIn("score_transicao", score)
        self.assertGreaterEqual(float(score["score_transicao"]), 0.0)
        self.assertLessEqual(float(score["score_transicao"]), 100.0)

    def test_resolve_exhaustive_weights_normalizes_custom_weights(self) -> None:
        weights = resolve_exhaustive_weights({"estatistico": 2.0, "historico": 1.0})

        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        self.assertGreater(weights["estatistico"], weights["historico"])

    def test_decision_layer_generates_single_prediction(self) -> None:
        rows = []
        for idx in range(1, 14):
            rows.append(normalize_contest(payload_with_dezenas(idx, cyclic_dezenas(idx))))

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            summary = build_single_prediction(
                pd.DataFrame(rows),
                existing_candidates=None,
                seed=123,
                candidate_pool=100,
                top_games=5,
                generations=1,
                population=8,
                draw_hour=20,
                draw_minute=0,
                engine="exaustivo",
                exhaustive_limit=200,
                weight_profile="padrao_atual",
                prediction_csv_path=base / "single.csv",
                report_path=base / "single.md",
                excel_path=base / "single.xlsx",
            )

            self.assertTrue((base / "single.csv").exists())
            self.assertTrue((base / "single.md").exists())
            self.assertTrue((base / "single.xlsx").exists())
            self.assertEqual(len(parse_numbers(summary.jogo_unico)), 15)

    def test_exhaustive_single_backtest_ablation_and_tuning(self) -> None:
        rows = []
        for idx in range(1, 14):
            rows.append(normalize_contest(payload_with_dezenas(idx, cyclic_dezenas(idx))))
        concursos = pd.DataFrame(rows)

        results, summary = run_exhaustive_single_backtest(
            concursos,
            n_eval=2,
            min_history=10,
            top_games=5,
            exhaustive_limit=120,
            weight_profile="padrao_atual",
        )
        self.assertEqual(int(results["concurso_previsto"].nunique()), 2)
        self.assertEqual(len(summary), 1)
        self.assertIn("media_acertos_jogo_unico", summary.columns)

        ablation_results, ablation_summary = run_ablation_test(
            concursos,
            n_eval=1,
            min_history=10,
            top_games=5,
            exhaustive_limit=80,
        )
        self.assertIn("completo", set(ablation_results["weight_profile"]))
        self.assertIn("delta_media_vs_completo", ablation_summary.columns)

        tuning_results, tuning_summary, best_payload = run_weight_tuning(
            concursos,
            n_eval=1,
            min_history=10,
            top_games=5,
            exhaustive_limit=80,
            profiles=["padrao_atual", "contexto_forte"],
        )
        self.assertEqual(set(tuning_results["weight_profile"]), {"padrao_atual", "contexto_forte"})
        self.assertIn("ranking_profile", tuning_summary.columns)
        self.assertIn(best_payload["best_profile"], {"padrao_atual", "contexto_forte"})
        self.assertAlmostEqual(sum(weights_for_profile(best_payload["best_profile"]).values()), 1.0, places=6)

    def test_select_final_games_returns_two_distinct_games(self) -> None:
        candidates = pd.DataFrame(
            [
                {"nums": "01 02 03 04 05 06 07 08 09 10 11 12 13 14 15", "score_final": 99.0, "metodo": "a"},
                {"nums": "01 02 03 04 05 06 07 08 09 16 17 18 19 20 21", "score_final": 98.0, "metodo": "b"},
                {"nums": "01 02 03 04 05 06 07 08 09 10 11 12 13 14 15", "score_final": 97.0, "metodo": "c"},
            ]
        )
        final_games = select_final_games(candidates, max_overlap=10)

        self.assertEqual(len(final_games), 2)
        self.assertNotEqual(final_games.loc[0, "nums"], final_games.loc[1, "nums"])
        self.assertLessEqual(int(final_games.loc[1, "overlap_com_jogo_1"]), 10)

    def test_select_final_games_uses_portfolio_score_for_second_game(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "nums": "01 02 03 04 05 06 07 08 09 10 11 12 13 14 15",
                    "score_final": 99.0,
                    "score_contextual": 70.0,
                    "score_transicao": 70.0,
                    "score_estatistico": 99.0,
                    "score_historico": 99.0,
                    "score_atraso": 99.0,
                    "score_combinatorio": 99.0,
                    "score_cenarios": 99.0,
                    "score_contrarian": 99.0,
                },
                {
                    "nums": "01 02 03 04 05 06 07 08 16 17 18 19 20 21 22",
                    "score_final": 95.0,
                    "score_contextual": 30.0,
                    "score_transicao": 30.0,
                    "score_estatistico": 95.0,
                    "score_historico": 95.0,
                    "score_atraso": 95.0,
                    "score_combinatorio": 95.0,
                    "score_cenarios": 95.0,
                    "score_contrarian": 95.0,
                },
                {
                    "nums": "01 02 03 04 05 06 07 08 18 19 20 21 22 23 24",
                    "score_final": 94.0,
                    "score_contextual": 100.0,
                    "score_transicao": 100.0,
                    "score_estatistico": 95.0,
                    "score_historico": 95.0,
                    "score_atraso": 95.0,
                    "score_combinatorio": 95.0,
                    "score_cenarios": 95.0,
                    "score_contrarian": 95.0,
                },
            ]
        )

        final_games = select_final_games(candidates, max_overlap=8)

        self.assertEqual(final_games.loc[1, "nums"], "01 02 03 04 05 06 07 08 18 19 20 21 22 23 24")
        self.assertIn("score_portfolio_jogo_2", final_games.columns)
        self.assertIn("score_diversidade_jogo_2", final_games.columns)
        self.assertEqual(final_games.loc[1, "criterio_selecao"], "portfolio_inteligente_overlap<=8")

    def test_decision_guard_marks_risk_numbers_and_enriches_candidates(self) -> None:
        candidates = pd.DataFrame(
            [
                {"rank": 1, "nums": "01 02 03 04 05 06 07 08 09 10 11 12 13 14 15", "score_final": 99.0, "score_contextual": 65.0, "score_transicao": 80.0},
                {"rank": 2, "nums": "01 02 03 04 05 06 07 08 09 10 11 12 13 14 16", "score_final": 98.8, "score_contextual": 64.0, "score_transicao": 81.0},
                {"rank": 3, "nums": "01 02 03 04 05 06 07 08 09 10 11 12 13 14 17", "score_final": 98.6, "score_contextual": 63.0, "score_transicao": 82.0},
                {"rank": 4, "nums": "01 02 03 04 05 06 07 08 09 10 11 12 13 14 18", "score_final": 98.4, "score_contextual": 62.0, "score_transicao": 83.0},
                {"rank": 5, "nums": "01 02 03 04 05 06 07 08 09 10 11 12 13 14 19", "score_final": 98.2, "score_contextual": 61.0, "score_transicao": 84.0},
            ]
        )

        guard_table = build_number_guard_table(candidates, consensus_top=3)
        enriched = enrich_candidates_with_decision_guard(candidates, consensus_top=3)

        self.assertIn("score_decisao_protegida", enriched.columns)
        self.assertIn("score_contexto_protegido", enriched.columns)
        self.assertIn("score_cobertura_risco_falso_negativo", enriched.columns)
        self.assertTrue((guard_table["categoria_guarda"] == "risco_falso_negativo").any())
        self.assertGreaterEqual(float(enriched["score_decisao_protegida"].max()), 0.0)

    def test_mandel_strategy_builds_plan_and_reduced_closure(self) -> None:
        candidates = pd.DataFrame(
            [
                {"rank": 1, "nums": "01 02 03 04 05 06 07 08 09 10 11 12 13 14 15", "score_final": 99.0, "score_contextual": 65.0, "score_transicao": 80.0},
                {"rank": 2, "nums": "01 02 03 04 05 06 07 08 09 10 11 12 13 14 16", "score_final": 98.8, "score_contextual": 64.0, "score_transicao": 81.0},
                {"rank": 3, "nums": "01 02 03 04 05 06 07 08 09 10 11 12 13 14 17", "score_final": 98.6, "score_contextual": 63.0, "score_transicao": 82.0},
            ]
        )

        universe = choose_strategy_universe(candidates, universe_size=17)
        plan = build_plan_table(candidates)
        games, coverage_pct, complete = greedy_reduced_closure(universe, guarantee_hits=14, max_games=20)

        self.assertEqual(len(universe), 17)
        self.assertIn("custo_desdobramento_completo", plan.columns)
        self.assertGreater(len(games), 0)
        self.assertGreaterEqual(coverage_pct, 0.0)
        self.assertTrue(complete)

    def test_mandel_strategy_writes_outputs(self) -> None:
        rows = []
        for idx in range(1, 14):
            rows.append(normalize_contest(payload_with_dezenas(idx, cyclic_dezenas(idx))))
        candidates = pd.DataFrame(
            [
                {"rank": 1, "nums": "01 02 03 04 05 06 07 08 09 10 11 12 13 14 15", "score_final": 99.0, "score_contextual": 65.0, "score_transicao": 80.0},
                {"rank": 2, "nums": "01 02 03 04 05 06 07 08 09 10 11 12 13 14 16", "score_final": 98.8, "score_contextual": 64.0, "score_transicao": 81.0},
                {"rank": 3, "nums": "01 02 03 04 05 06 07 08 09 10 11 12 13 14 17", "score_final": 98.6, "score_contextual": 63.0, "score_transicao": 82.0},
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            summary = run_mandel_strategy(
                pd.DataFrame(rows),
                candidates,
                universe_size=17,
                guarantee_hits=14,
                max_reduced_games=20,
                plan_csv_path=base / "plan.csv",
                games_csv_path=base / "games.csv",
                report_path=base / "report.md",
                excel_path=base / "mandel.xlsx",
            )

            self.assertTrue((base / "plan.csv").exists())
            self.assertTrue((base / "games.csv").exists())
            self.assertTrue((base / "report.md").exists())
            self.assertTrue((base / "mandel.xlsx").exists())
            self.assertEqual(summary.tamanho_universo, 17)

    def test_web_interface_html_contains_expected_controls(self) -> None:
        html = _html_page()
        self.assertIn("Lotofacil Analytics", html)
        self.assertIn("/api/status", html)
        self.assertIn("/api/calibration/status", html)
        self.assertIn("/api/transitions", html)
        self.assertIn("/api/climate", html)
        self.assertIn("/api/predict", html)
        self.assertIn("/api/predict-single", html)
        self.assertIn("/api/mandel", html)
        self.assertIn("/report", html)
        self.assertIn("/mandel-report", html)
        self.assertIn("Comparação visual dos scores", html)
        self.assertIn("Score climático", html)
        self.assertIn("Temporal profundo", html)
        self.assertIn("score_portfolio_jogo_2", html)
        self.assertIn("Decisão protegida", html)
        self.assertIn("Anti-falso-negativo", html)
        self.assertIn("Calibração 24/7", html)

    def test_generate_games_balanceado_returns_requested_quantity(self) -> None:
        rows = []
        for idx in range(1, 8):
            start = idx
            dezenas = [f"{n:02d}" for n in range(start, start + 15)]
            rows.append(normalize_contest(payload_with_dezenas(idx, dezenas)))

        games = generate_games(
            pd.DataFrame(rows),
            method="balanceado_basico",
            qty=3,
            seed=123,
            window=2,
            candidates=100,
            candidate_pool=200,
            generations=2,
            population=10,
        )

        self.assertEqual(len(games), 3)
        for nums_text in games["nums"].tolist():
            nums = [int(part) for part in nums_text.split()]
            self.assertEqual(len(nums), 15)
            self.assertEqual(len(set(nums)), 15)
            self.assertTrue(all(1 <= n <= 25 for n in nums))

    def test_final_score_backtest_compares_model_with_random_baseline(self) -> None:
        rows = []
        for idx in range(1, 14):
            rows.append(normalize_contest(payload_with_dezenas(idx, cyclic_dezenas(idx))))

        results, summary = run_final_score_backtest(
            pd.DataFrame(rows),
            n_eval=2,
            min_history=10,
            seed=123,
            candidate_pool=80,
            top_games=10,
            generations=1,
            population=8,
            max_overlap=10,
        )

        self.assertEqual(int(results["concurso_previsto"].nunique()), 2)
        self.assertEqual(set(results["modelo_nome"]), {"ensemble_score_v2", "baseline_2_jogos_aleatorios"})
        self.assertEqual(set(summary["modelo_nome"]), set(results["modelo_nome"]))
        self.assertTrue((results["melhor_acerto_entre_2"] >= 0).all())


if __name__ == "__main__":
    unittest.main()
