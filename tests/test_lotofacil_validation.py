from __future__ import annotations

import unittest

import pandas as pd

from lotofacil_analytics.combinacoes import build_combinacoes_features, build_combinacoes_outputs
from lotofacil_analytics.dezenas_history import build_dezenas_historico, build_dezenas_long
from lotofacil_analytics.features_base import build_base_features
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


if __name__ == "__main__":
    unittest.main()
