from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEZENAS = [f"d{i:02d}" for i in range(1, 16)]
ORDEM_SORTEIO = [f"ordem_{i:02d}" for i in range(1, 16)]


def parse_caixa_date(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    text = str(value).strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"Data invalida: {value!r}")


def parse_int_list(values: Iterable[Any], *, expected: int, min_value: int, max_value: int) -> List[int]:
    nums: List[int] = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        nums.append(int(text))
    if len(nums) != expected:
        raise ValueError(f"Esperado {expected} dezenas, recebido {len(nums)}: {nums}")
    if len(set(nums)) != expected:
        raise ValueError(f"Dezenas repetidas: {nums}")
    invalid = [n for n in nums if n < min_value or n > max_value]
    if invalid:
        raise ValueError(f"Dezenas fora de {min_value}..{max_value}: {invalid}")
    return nums


def split_city_uf(value: Any) -> Tuple[Optional[str], Optional[str]]:
    if value in (None, ""):
        return None, None
    text = str(value).strip()
    if "," not in text:
        return text or None, None
    city, uf = text.rsplit(",", 1)
    return city.strip() or None, uf.strip() or None


def rateio_map(payload: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    result: Dict[int, Dict[str, Any]] = {}
    for item in payload.get("listaRateioPremio") or []:
        try:
            faixa = int(item.get("faixa"))
        except (TypeError, ValueError):
            continue
        result[faixa] = item
    return result


def normalize_contest(payload: Dict[str, Any], *, raw_json_file: str | None = None) -> Dict[str, Any]:
    concurso = int(payload["numero"])
    dezenas = sorted(parse_int_list(payload.get("listaDezenas") or [], expected=15, min_value=1, max_value=25))
    ordem = parse_int_list(
        payload.get("dezenasSorteadasOrdemSorteio") or payload.get("listaDezenas") or [],
        expected=15,
        min_value=1,
        max_value=25,
    )
    cidade_sorteio, uf_sorteio = split_city_uf(payload.get("nomeMunicipioUFSorteio"))
    rateios = rateio_map(payload)

    record: Dict[str, Any] = {
        "concurso": concurso,
        "data_sorteio": parse_caixa_date(payload.get("dataApuracao")),
        "data_proximo_concurso": parse_caixa_date(payload.get("dataProximoConcurso")),
        "tipo_jogo": payload.get("tipoJogo"),
        "acumulado": bool(payload.get("acumulado")) if payload.get("acumulado") is not None else None,
        "local_sorteio": payload.get("localSorteio"),
        "cidade_sorteio": cidade_sorteio,
        "uf_sorteio": uf_sorteio,
        "indicador_concurso_especial": payload.get("indicadorConcursoEspecial"),
        "valor_arrecadado": payload.get("valorArrecadado"),
        "valor_acumulado_proximo_concurso": payload.get("valorAcumuladoProximoConcurso"),
        "valor_estimado_proximo_concurso": payload.get("valorEstimadoProximoConcurso"),
        "valor_acumulado_concurso_especial": payload.get("valorAcumuladoConcursoEspecial"),
        "valor_total_premio_faixa_um": payload.get("valorTotalPremioFaixaUm"),
        "observacao": payload.get("observacao"),
        "raw_json_file": raw_json_file,
    }

    for idx, dezena in enumerate(dezenas, start=1):
        record[f"d{idx:02d}"] = dezena
    for idx, dezena in enumerate(ordem, start=1):
        record[f"ordem_{idx:02d}"] = dezena

    for acertos, faixa in [(15, 1), (14, 2), (13, 3), (12, 4), (11, 5)]:
        item = rateios.get(faixa, {})
        record[f"ganhadores_{acertos}"] = item.get("numeroDeGanhadores")
        record[f"premio_{acertos}"] = item.get("valorPremio")

    return record


def column_order() -> List[str]:
    base = [
        "concurso",
        "data_sorteio",
        "data_proximo_concurso",
        "tipo_jogo",
        "acumulado",
        "local_sorteio",
        "cidade_sorteio",
        "uf_sorteio",
        "indicador_concurso_especial",
    ]
    money = [
        "valor_arrecadado",
        "valor_acumulado_proximo_concurso",
        "valor_estimado_proximo_concurso",
        "valor_acumulado_concurso_especial",
        "valor_total_premio_faixa_um",
    ]
    prizes: List[str] = []
    for acertos in [15, 14, 13, 12, 11]:
        prizes.extend([f"ganhadores_{acertos}", f"premio_{acertos}"])
    return base + DEZENAS + ORDEM_SORTEIO + prizes + money + ["observacao", "raw_json_file"]
