from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List

from .normalize import DEZENAS, ORDEM_SORTEIO


class DataValidationError(ValueError):
    pass


def _require_date(value: Any, *, field: str, concurso: int) -> None:
    if value in (None, ""):
        raise DataValidationError(f"Concurso {concurso}: campo {field} vazio.")
    try:
        datetime.strptime(str(value), "%Y-%m-%d")
    except ValueError as exc:
        raise DataValidationError(f"Concurso {concurso}: campo {field} invalido: {value!r}") from exc


def _validate_numbers(values: List[Any], *, expected: int, min_value: int, max_value: int, label: str, concurso: int) -> None:
    try:
        nums = [int(v) for v in values]
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"Concurso {concurso}: {label} contem valor nao inteiro.") from exc
    if len(nums) != expected:
        raise DataValidationError(f"Concurso {concurso}: {label} deve ter {expected} dezenas.")
    if len(set(nums)) != expected:
        raise DataValidationError(f"Concurso {concurso}: {label} tem dezenas repetidas: {nums}")
    invalid = [n for n in nums if n < min_value or n > max_value]
    if invalid:
        raise DataValidationError(f"Concurso {concurso}: {label} fora de {min_value}..{max_value}: {invalid}")


def validate_contest_record(record: Dict[str, Any]) -> None:
    try:
        concurso = int(record["concurso"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataValidationError("Registro sem concurso numerico valido.") from exc
    if concurso < 1:
        raise DataValidationError(f"Concurso invalido: {concurso}")

    _require_date(record.get("data_sorteio"), field="data_sorteio", concurso=concurso)
    _validate_numbers([record.get(c) for c in DEZENAS], expected=15, min_value=1, max_value=25, label="dezenas", concurso=concurso)
    _validate_numbers(
        [record.get(c) for c in ORDEM_SORTEIO],
        expected=15,
        min_value=1,
        max_value=25,
        label="ordem_sorteio",
        concurso=concurso,
    )


def validate_dataset(records: Iterable[Dict[str, Any]], *, require_contiguous: bool = True) -> List[Dict[str, Any]]:
    rows = sorted(list(records), key=lambda r: int(r["concurso"]))
    if not rows:
        raise DataValidationError("Nenhum concurso encontrado.")

    seen: set[int] = set()
    for record in rows:
        validate_contest_record(record)
        concurso = int(record["concurso"])
        if concurso in seen:
            raise DataValidationError(f"Concurso duplicado: {concurso}")
        seen.add(concurso)

    if require_contiguous:
        expected = list(range(int(rows[0]["concurso"]), int(rows[-1]["concurso"]) + 1))
        found = [int(r["concurso"]) for r in rows]
        if found != expected:
            missing = sorted(set(expected) - set(found))
            raise DataValidationError(f"Historico com concursos ausentes: {missing[:20]}")

    return rows
