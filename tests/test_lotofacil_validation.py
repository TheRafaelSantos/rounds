from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
