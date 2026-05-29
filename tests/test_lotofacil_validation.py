from __future__ import annotations

import unittest

import pandas as pd

from lotofacil_analytics.auditoria import build_auditoria
from lotofacil_analytics.backtest_lotofacil import compute_hits, run_backtest
from lotofacil_analytics.combinacoes import build_combinacoes_features, build_combinacoes_outputs
from lotofacil_analytics.dezenas_history import build_dezenas_historico, build_dezenas_long
from lotofacil_analytics.features_base import build_base_features
from lotofacil_analytics.ml_temporal import run_ml_temporal
from lotofacil_analytics.optimizer import build_optimized_candidates, score_candidate
from lotofacil_analytics.predictor import select_final_games
from lotofacil_analytics.normalize import normalize_contest
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


if __name__ == "__main__":
    unittest.main()
