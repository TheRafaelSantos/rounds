from __future__ import annotations

import math
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, time, timezone
from typing import Dict, List, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from .normalize import DEZENAS


BRASILIA_TZ = ZoneInfo("America/Sao_Paulo")
SYNODIC_MONTH_DAYS = 29.530588853
KNOWN_NEW_MOON_UTC = datetime(2000, 1, 6, 18, 14, tzinfo=timezone.utc)
DIA_SEMANA_NOMES = {
    1: "segunda-feira",
    2: "terca-feira",
    3: "quarta-feira",
    4: "quinta-feira",
    5: "sexta-feira",
    6: "sabado",
    7: "domingo",
}


@dataclass(frozen=True)
class TargetContext:
    concurso_alvo: int
    data_proximo_concurso: str
    horario_brasilia_assumido: str
    datetime_brasilia: str
    dia_semana_numero: int
    dia_semana_nome: str
    mes: int
    trimestre: int
    semestre: int
    estacao_do_ano: str
    fase_lua: str
    idade_lua: float
    iluminacao_lua_percentual: float
    numerologia_data_raiz: int
    numerologia_concurso_raiz: int
    numerologia_dia_mes_raiz: int
    local_sorteio_assumido: str
    cidade_sorteio_assumida: str
    uf_sorteio_assumida: str
    bairro_sorteio_assumido: str
    observacao_horario: str
    observacao_localidade: str

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ContextModel:
    target: TargetContext
    sample_sizes: Dict[str, int]
    counters: Dict[str, Counter[int]]


def _nums_from_row(row: pd.Series) -> List[int]:
    return sorted(int(row[col]) for col in DEZENAS)


def digital_root(value: int) -> int:
    value = abs(int(value))
    if value == 0:
        return 0
    return 1 + ((value - 1) % 9)


def normalize_context_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = "".join(ch.upper() if ch.isalnum() else " " for ch in text)
    return " ".join(text.split())


def estacao_do_ano(date: pd.Timestamp) -> str:
    month_day = (int(date.month), int(date.day))
    if month_day >= (12, 21) or month_day < (3, 20):
        return "verao"
    if month_day < (6, 21):
        return "outono"
    if month_day < (9, 23):
        return "inverno"
    return "primavera"


def moon_phase(draw_datetime_brasilia: datetime) -> Dict[str, object]:
    dt_utc = draw_datetime_brasilia.astimezone(timezone.utc)
    age = ((dt_utc - KNOWN_NEW_MOON_UTC).total_seconds() / 86400.0) % SYNODIC_MONTH_DAYS
    illumination = (1.0 - math.cos(2.0 * math.pi * age / SYNODIC_MONTH_DAYS)) / 2.0 * 100.0
    if age < 1.84566:
        name = "lua_nova"
    elif age < 5.53699:
        name = "lua_crescente_inicial"
    elif age < 9.22831:
        name = "quarto_crescente"
    elif age < 12.91963:
        name = "gibosa_crescente"
    elif age < 16.61096:
        name = "lua_cheia"
    elif age < 20.30228:
        name = "gibosa_minguante"
    elif age < 23.99361:
        name = "quarto_minguante"
    elif age < 27.68493:
        name = "lua_minguante_final"
    else:
        name = "lua_nova"
    return {
        "fase_lua": name,
        "idade_lua": round(float(age), 6),
        "iluminacao_lua_percentual": round(float(illumination), 6),
    }


def _target_date_from_history(concursos: pd.DataFrame) -> pd.Timestamp:
    last = concursos.copy().sort_values("concurso").iloc[-1]
    next_date = pd.to_datetime(last.get("data_proximo_concurso"), errors="coerce")
    if pd.isna(next_date):
        last_date = pd.to_datetime(last["data_sorteio"], errors="coerce")
        if pd.isna(last_date):
            raise ValueError("Nao foi possivel determinar a data do proximo concurso.")
        next_date = last_date + pd.Timedelta(days=1)
    return pd.Timestamp(next_date)


def build_target_context(concursos: pd.DataFrame, *, draw_hour: int = 20, draw_minute: int = 0) -> TargetContext:
    if concursos.empty:
        raise ValueError("Historico local nao encontrado.")
    df = concursos.copy().sort_values("concurso").reset_index(drop=True)
    last = df.iloc[-1]
    target_date = _target_date_from_history(df)
    concurso_alvo = int(last["concurso"]) + 1
    draw_time = time(hour=int(draw_hour), minute=int(draw_minute))
    draw_datetime = datetime.combine(target_date.date(), draw_time, tzinfo=BRASILIA_TZ)
    moon = moon_phase(draw_datetime)
    weekday = int(target_date.isoweekday())
    local = normalize_context_text(last.get("local_sorteio", ""))
    cidade = normalize_context_text(last.get("cidade_sorteio", ""))
    uf = normalize_context_text(last.get("uf_sorteio", ""))
    bairro = normalize_context_text(last.get("bairro_sorteio", "")) if "bairro_sorteio" in last.index else ""
    return TargetContext(
        concurso_alvo=concurso_alvo,
        data_proximo_concurso=target_date.date().isoformat(),
        horario_brasilia_assumido=f"{int(draw_hour):02d}:{int(draw_minute):02d}",
        datetime_brasilia=draw_datetime.isoformat(),
        dia_semana_numero=weekday,
        dia_semana_nome=DIA_SEMANA_NOMES[weekday],
        mes=int(target_date.month),
        trimestre=int((target_date.month - 1) // 3 + 1),
        semestre=int((target_date.month - 1) // 6 + 1),
        estacao_do_ano=estacao_do_ano(target_date),
        fase_lua=str(moon["fase_lua"]),
        idade_lua=float(moon["idade_lua"]),
        iluminacao_lua_percentual=float(moon["iluminacao_lua_percentual"]),
        numerologia_data_raiz=digital_root(int(target_date.strftime("%Y%m%d"))),
        numerologia_concurso_raiz=digital_root(concurso_alvo),
        numerologia_dia_mes_raiz=digital_root(int(target_date.day) + int(target_date.month)),
        local_sorteio_assumido=local,
        cidade_sorteio_assumida=cidade,
        uf_sorteio_assumida=uf,
        bairro_sorteio_assumido=bairro,
        observacao_horario="A API da CAIXA informa a data do proximo concurso, mas nao o horario; foi usado horario de Brasilia configuravel.",
        observacao_localidade="A API local informa a localidade do sorteio realizado, mas nao garante localidade futura; para o proximo concurso foi usada a localidade do ultimo concurso disponivel como premissa auditavel.",
    )


def build_context_model(concursos: pd.DataFrame, *, draw_hour: int = 20, draw_minute: int = 0) -> ContextModel:
    target = build_target_context(concursos, draw_hour=draw_hour, draw_minute=draw_minute)
    counters: Dict[str, Counter[int]] = defaultdict(Counter)
    sample_sizes: Dict[str, int] = defaultdict(int)
    df = concursos.copy().sort_values("concurso").reset_index(drop=True)

    for _, row in df.iterrows():
        draw_date = pd.to_datetime(row["data_sorteio"], errors="coerce")
        if pd.isna(draw_date):
            continue
        draw_datetime = datetime.combine(draw_date.date(), time(hour=int(draw_hour), minute=int(draw_minute)), tzinfo=BRASILIA_TZ)
        moon = moon_phase(draw_datetime)
        keys = [
            f"weekday:{int(draw_date.isoweekday())}",
            f"month:{int(draw_date.month)}",
            f"quarter:{int((draw_date.month - 1) // 3 + 1)}",
            f"semester:{int((draw_date.month - 1) // 6 + 1)}",
            f"season:{estacao_do_ano(draw_date)}",
            f"moon:{moon['fase_lua']}",
            f"numerology_date:{digital_root(int(draw_date.strftime('%Y%m%d')))}",
            f"numerology_concurso:{digital_root(int(row['concurso']))}",
            f"numerology_day_month:{digital_root(int(draw_date.day) + int(draw_date.month))}",
        ]
        local = normalize_context_text(row.get("local_sorteio", ""))
        cidade = normalize_context_text(row.get("cidade_sorteio", ""))
        uf = normalize_context_text(row.get("uf_sorteio", ""))
        bairro = normalize_context_text(row.get("bairro_sorteio", "")) if "bairro_sorteio" in row.index else ""
        if local:
            keys.append(f"local:{local}")
        if cidade:
            keys.append(f"cidade:{cidade}")
        if uf:
            keys.append(f"uf:{uf}")
        if bairro:
            keys.append(f"bairro:{bairro}")
        if cidade and uf:
            keys.append(f"cidade_uf:{cidade}|{uf}")
        nums = _nums_from_row(row)
        for key in keys:
            sample_sizes[key] += 1
            counters[key].update(nums)

    return ContextModel(target=target, sample_sizes=dict(sample_sizes), counters=dict(counters))


def _frequency_score(nums: Sequence[int], *, counter: Counter[int], sample_size: int) -> float:
    if sample_size <= 0:
        return 50.0
    expected = sample_size * 15.0 / 25.0
    avg_selected = sum(counter.get(int(n), 0) for n in nums) / 15.0
    raw = 50.0 + (avg_selected - expected) * 7.0
    shrink = min(1.0, sample_size / 50.0)
    return round(max(0.0, min(100.0, 50.0 + (raw - 50.0) * shrink)), 6)


def _numerology_score(nums: Sequence[int], target: TargetContext) -> float:
    target_roots = {
        target.numerologia_data_raiz,
        target.numerologia_concurso_raiz,
        target.numerologia_dia_mes_raiz,
        digital_root(target.dia_semana_numero),
        digital_root(target.mes),
    }
    matches = sum(1 for n in nums if digital_root(int(n)) in target_roots)
    expected = 15.0 * len(target_roots) / 9.0
    raw = 50.0 + (matches - expected) * 8.0
    return round(max(0.0, min(100.0, raw)), 6)


def score_contextual_candidate(nums: Sequence[int], model: ContextModel) -> Dict[str, object]:
    target = model.target
    keys = {
        "score_dia_semana": f"weekday:{target.dia_semana_numero}",
        "score_mes": f"month:{target.mes}",
        "score_trimestre": f"quarter:{target.trimestre}",
        "score_semestre": f"semester:{target.semestre}",
        "score_estacao": f"season:{target.estacao_do_ano}",
        "score_lua": f"moon:{target.fase_lua}",
        "score_numerologia_data": f"numerology_date:{target.numerologia_data_raiz}",
        "score_numerologia_concurso": f"numerology_concurso:{target.numerologia_concurso_raiz}",
        "score_numerologia_dia_mes": f"numerology_day_month:{target.numerologia_dia_mes_raiz}",
    }
    if target.local_sorteio_assumido:
        keys["score_local_sorteio"] = f"local:{target.local_sorteio_assumido}"
    if target.cidade_sorteio_assumida:
        keys["score_cidade_sorteio"] = f"cidade:{target.cidade_sorteio_assumida}"
    if target.uf_sorteio_assumida:
        keys["score_uf_sorteio"] = f"uf:{target.uf_sorteio_assumida}"
    if target.bairro_sorteio_assumido:
        keys["score_bairro_sorteio"] = f"bairro:{target.bairro_sorteio_assumido}"
    if target.cidade_sorteio_assumida and target.uf_sorteio_assumida:
        keys["score_cidade_uf_sorteio"] = f"cidade_uf:{target.cidade_sorteio_assumida}|{target.uf_sorteio_assumida}"
    scores: Dict[str, float] = {}
    for score_name, key in keys.items():
        scores[score_name] = _frequency_score(
            nums,
            counter=model.counters.get(key, Counter()),
            sample_size=int(model.sample_sizes.get(key, 0)),
        )
    scores["score_numerologia"] = _numerology_score(nums, target)
    temporal_score = round(
        0.18 * scores["score_dia_semana"]
        + 0.12 * scores["score_mes"]
        + 0.10 * scores["score_trimestre"]
        + 0.08 * scores["score_semestre"]
        + 0.14 * scores["score_estacao"]
        + 0.18 * scores["score_lua"]
        + 0.06 * scores["score_numerologia_data"]
        + 0.05 * scores["score_numerologia_concurso"]
        + 0.04 * scores["score_numerologia_dia_mes"]
        + 0.05 * scores["score_numerologia"],
        6,
    )
    locality_values = [
        scores[name]
        for name in [
            "score_local_sorteio",
            "score_cidade_sorteio",
            "score_uf_sorteio",
            "score_bairro_sorteio",
            "score_cidade_uf_sorteio",
        ]
        if name in scores
    ]
    locality_score = round(float(sum(locality_values) / len(locality_values)), 6) if locality_values else 50.0
    score_contextual = round(0.82 * temporal_score + 0.18 * locality_score, 6)
    return {
        **scores,
        "score_contextual_temporal": temporal_score,
        "score_localidade": locality_score,
        "score_contextual": score_contextual,
        "contexto_data_proximo_concurso": target.data_proximo_concurso,
        "contexto_horario_brasilia": target.horario_brasilia_assumido,
        "contexto_dia_semana": target.dia_semana_nome,
        "contexto_estacao": target.estacao_do_ano,
        "contexto_fase_lua": target.fase_lua,
        "contexto_idade_lua": target.idade_lua,
        "contexto_iluminacao_lua_percentual": target.iluminacao_lua_percentual,
        "contexto_numerologia_data_raiz": target.numerologia_data_raiz,
        "contexto_numerologia_concurso_raiz": target.numerologia_concurso_raiz,
        "contexto_numerologia_dia_mes_raiz": target.numerologia_dia_mes_raiz,
        "contexto_local_sorteio": target.local_sorteio_assumido,
        "contexto_cidade_sorteio": target.cidade_sorteio_assumida,
        "contexto_uf_sorteio": target.uf_sorteio_assumida,
        "contexto_bairro_sorteio": target.bairro_sorteio_assumido,
        "contexto_observacao_localidade": target.observacao_localidade,
    }
