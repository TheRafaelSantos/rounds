from __future__ import annotations

import json
import logging
import time as time_module
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pandas as pd
import requests

from .context_features import BRASILIA_TZ, build_target_context, normalize_context_text


OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
CLIMATE_SOURCE = "open_meteo_historical_forecast_cache"
CLIMATE_VARIABLES = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "surface_pressure",
    "precipitation",
]
UF_ADMIN1 = {
    "AC": "ACRE",
    "AL": "ALAGOAS",
    "AP": "AMAPA",
    "AM": "AMAZONAS",
    "BA": "BAHIA",
    "CE": "CEARA",
    "DF": "DISTRITO FEDERAL",
    "ES": "ESPIRITO SANTO",
    "GO": "GOIAS",
    "MA": "MARANHAO",
    "MT": "MATO GROSSO",
    "MS": "MATO GROSSO DO SUL",
    "MG": "MINAS GERAIS",
    "PA": "PARA",
    "PB": "PARAIBA",
    "PR": "PARANA",
    "PE": "PERNAMBUCO",
    "PI": "PIAUI",
    "RJ": "RIO DE JANEIRO",
    "RN": "RIO GRANDE DO NORTE",
    "RS": "RIO GRANDE DO SUL",
    "RO": "RONDONIA",
    "RR": "RORAIMA",
    "SC": "SANTA CATARINA",
    "SP": "SAO PAULO",
    "SE": "SERGIPE",
    "TO": "TOCANTINS",
}


@dataclass(frozen=True)
class ClimateSummary:
    rows: int
    locations_processed: int
    locations_total: int
    geocoded_locations: int
    failed_locations: int
    csv_path: str
    excel_path: str

    def to_console(self) -> str:
        return "\n".join(
            [
                "",
                "Resumo Lotofacil Analytics - Camada Climática",
                f"Linhas climáticas: {self.rows}",
                f"Localidades processadas: {self.locations_processed}/{self.locations_total}",
                f"Localidades geocodificadas: {self.geocoded_locations}",
                f"Localidades com falha: {self.failed_locations}",
                f"CSV: {self.csv_path}",
                f"Excel: {self.excel_path}",
                "Mensagem: Clima histórico gerado para uso experimental no motor principal.",
            ]
        )


def _ascii_slug(value: object) -> str:
    text = normalize_context_text(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return "_".join(text.lower().split()) or "indefinido"


def _cache_file(cache_dir: Path, prefix: str, *parts: object) -> Path:
    safe = "__".join(_ascii_slug(part) for part in parts)
    return cache_dir / f"{prefix}__{safe}.json"


def _load_json(path: Path) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def _request_json(
    url: str,
    *,
    params: Mapping[str, object],
    timeout_seconds: float,
    retries: int,
    sleep_seconds: float,
) -> Mapping[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max(1, int(retries))):
        try:
            response = requests.get(url, params=params, timeout=float(timeout_seconds))
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                return payload
            raise ValueError("Resposta JSON nao e um objeto.")
        except Exception as exc:  # pragma: no cover - exercised by integration runs
            last_error = exc
            if attempt + 1 < max(1, int(retries)):
                time_module.sleep(max(0.0, float(sleep_seconds)))
    raise RuntimeError(f"Falha ao consultar Open-Meteo: {last_error}") from last_error


def _best_geocode_result(results: Sequence[Mapping[str, Any]], *, city: str, uf: str) -> Mapping[str, Any] | None:
    if not results:
        return None
    city_norm = normalize_context_text(city)
    uf_norm = normalize_context_text(uf)
    admin_expected = UF_ADMIN1.get(uf_norm, "")

    def score(item: Mapping[str, Any]) -> int:
        item_name = normalize_context_text(item.get("name", ""))
        item_admin = normalize_context_text(item.get("admin1", ""))
        item_country = normalize_context_text(item.get("country_code", ""))
        value = 0
        value += 100 if item_country == "BR" else 0
        value += 60 if item_name == city_norm else 0
        value += 35 if admin_expected and item_admin == admin_expected else 0
        value += 10 if city_norm and city_norm in item_name else 0
        return value

    return sorted(results, key=score, reverse=True)[0]


def geocode_city(
    *,
    city: str,
    uf: str,
    cache_dir: Path,
    timeout_seconds: float,
    retries: int,
    sleep_seconds: float,
    force: bool = False,
) -> Dict[str, object]:
    city_norm = normalize_context_text(city)
    uf_norm = normalize_context_text(uf)
    if not city_norm or not uf_norm:
        raise ValueError("Cidade/UF ausentes para geocodificacao climatica.")
    cache_path = _cache_file(cache_dir, "geocode", city_norm, uf_norm)
    cached = None if force else _load_json(cache_path)
    if cached:
        return dict(cached)

    payload = _request_json(
        OPEN_METEO_GEOCODING_URL,
        params={
            "name": city_norm,
            "count": 10,
            "language": "pt",
            "format": "json",
            "countryCode": "BR",
        },
        timeout_seconds=timeout_seconds,
        retries=retries,
        sleep_seconds=sleep_seconds,
    )
    result = _best_geocode_result(payload.get("results") or [], city=city_norm, uf=uf_norm)
    if not result:
        raise ValueError(f"Open-Meteo nao encontrou coordenadas para {city_norm}/{uf_norm}.")
    out = {
        "cidade": city_norm,
        "uf": uf_norm,
        "latitude": float(result["latitude"]),
        "longitude": float(result["longitude"]),
        "geocode_name": result.get("name", ""),
        "geocode_admin1": result.get("admin1", ""),
        "geocode_country_code": result.get("country_code", ""),
    }
    _save_json(cache_path, out)
    return out


def _weather_endpoint_for_range(start_date: str, end_date: str) -> str:
    today = datetime.now(BRASILIA_TZ).date()
    end = pd.to_datetime(end_date, errors="coerce")
    if not pd.isna(end) and end.date() >= today - timedelta(days=1):
        return OPEN_METEO_FORECAST_URL
    return OPEN_METEO_ARCHIVE_URL


def fetch_weather_range(
    *,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    cache_dir: Path,
    timeout_seconds: float,
    retries: int,
    sleep_seconds: float,
    force: bool = False,
) -> pd.DataFrame:
    endpoint = _weather_endpoint_for_range(start_date, end_date)
    cache_path = _cache_file(cache_dir, "weather", latitude, longitude, start_date, end_date, endpoint.rsplit("/", 1)[-1])
    payload = None if force else _load_json(cache_path)
    if payload is None:
        params: Dict[str, object] = {
            "latitude": round(float(latitude), 5),
            "longitude": round(float(longitude), 5),
            "hourly": ",".join(CLIMATE_VARIABLES),
            "timezone": "America/Sao_Paulo",
            "temperature_unit": "celsius",
            "precipitation_unit": "mm",
        }
        if endpoint == OPEN_METEO_FORECAST_URL:
            start = pd.to_datetime(start_date).date()
            end = pd.to_datetime(end_date).date()
            today = datetime.now(BRASILIA_TZ).date()
            params["forecast_days"] = max(1, min(16, (end - today).days + 1))
            params["past_days"] = max(0, min(92, (today - start).days))
        else:
            params["start_date"] = start_date
            params["end_date"] = end_date
        payload = _request_json(
            endpoint,
            params=params,
            timeout_seconds=timeout_seconds,
            retries=retries,
            sleep_seconds=sleep_seconds,
        )
        _save_json(cache_path, payload)

    hourly = payload.get("hourly") if isinstance(payload, Mapping) else None
    if not isinstance(hourly, Mapping) or "time" not in hourly:
        return pd.DataFrame()
    df = pd.DataFrame(hourly)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df.dropna(subset=["time"]).reset_index(drop=True)


def _temperature_band(value: object) -> str:
    value = _safe_float(value)
    if pd.isna(value):
        return "indisponivel"
    if value < 15:
        return "frio"
    if value < 22:
        return "ameno"
    if value < 28:
        return "quente"
    return "muito_quente"


def _humidity_band(value: object) -> str:
    value = _safe_float(value)
    if pd.isna(value):
        return "indisponivel"
    if value < 40:
        return "seco"
    if value < 70:
        return "normal"
    return "umido"


def _precipitation_band(value: object) -> str:
    value = _safe_float(value)
    if pd.isna(value):
        return "indisponivel"
    if value <= 0:
        return "sem_chuva"
    if value < 2:
        return "chuva_fraca"
    if value < 10:
        return "chuva_moderada"
    return "chuva_forte"


def _anomaly_band(value: object) -> str:
    value = _safe_float(value)
    if pd.isna(value):
        return "indisponivel"
    if value <= -3:
        return "mais_frio_que_media"
    if value >= 3:
        return "mais_quente_que_media"
    return "normal"


def _pressure_band(value: object, *, low: float | None, high: float | None) -> str:
    value = _safe_float(value)
    if pd.isna(value):
        return "indisponivel"
    if low is None or high is None:
        return "pressao_normal"
    if value < low:
        return "pressao_baixa_relativa"
    if value > high:
        return "pressao_alta_relativa"
    return "pressao_normal"


def _safe_float(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return numeric if not pd.isna(numeric) else float("nan")


def _hourly_at_draw_time(weather: pd.DataFrame, *, draw_hour: int, draw_minute: int) -> pd.DataFrame:
    if weather.empty or "time" not in weather.columns:
        return pd.DataFrame()
    out = weather.copy()
    out = out[
        (out["time"].dt.hour == int(draw_hour))
        & (out["time"].dt.minute == int(draw_minute))
    ].copy()
    out["data_sorteio"] = out["time"].dt.date.astype(str)
    return out


def _temperature_reference_map(hourly: pd.DataFrame) -> Dict[str, float]:
    if hourly.empty or "temperature_2m" not in hourly.columns:
        return {}
    df = hourly[["data_sorteio", "temperature_2m"]].copy()
    df["data_sorteio"] = pd.to_datetime(df["data_sorteio"], errors="coerce")
    df["temperature_2m"] = pd.to_numeric(df["temperature_2m"], errors="coerce")
    df = df.dropna(subset=["data_sorteio", "temperature_2m"]).sort_values("data_sorteio")
    if df.empty:
        return {}
    by_date = df.set_index("data_sorteio")["temperature_2m"]
    fallback = float(by_date.mean())
    refs: Dict[str, float] = {}
    for date in by_date.index:
        start = date - pd.Timedelta(days=30)
        recent = by_date[(by_date.index < date) & (by_date.index >= start)]
        refs[date.date().isoformat()] = round(float(recent.mean() if len(recent) else fallback), 6)
    return refs


def _finalize_climate_rows(rows: List[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["clima_surface_pressure"] = pd.to_numeric(df["clima_surface_pressure"], errors="coerce")
    df["clima_pressao_faixa"] = "indisponivel"
    group_cols = ["cidade_sorteio_normalizada", "uf_sorteio_normalizada"]
    for _, idxs in df.groupby(group_cols, dropna=False).groups.items():
        subset = df.loc[list(idxs), "clima_surface_pressure"].dropna()
        low = high = None
        if len(subset) >= 3:
            low = float(subset.quantile(0.33))
            high = float(subset.quantile(0.66))
        for idx in idxs:
            df.loc[idx, "clima_pressao_faixa"] = _pressure_band(df.loc[idx, "clima_surface_pressure"], low=low, high=high)
    df["clima_assinatura"] = (
        df["clima_temperatura_faixa"].astype(str)
        + "|"
        + df["clima_sensacao_faixa"].astype(str)
        + "|"
        + df["clima_umidade_faixa"].astype(str)
        + "|"
        + df["clima_pressao_faixa"].astype(str)
        + "|"
        + df["clima_chuva_faixa"].astype(str)
        + "|"
        + df["clima_anomalia_faixa"].astype(str)
    )
    return df.sort_values(["concurso"]).reset_index(drop=True)


def _build_rows_for_location(
    contests: pd.DataFrame,
    *,
    geocode: Mapping[str, object],
    weather: pd.DataFrame,
    draw_hour: int,
    draw_minute: int,
) -> List[Dict[str, object]]:
    hourly = _hourly_at_draw_time(weather, draw_hour=draw_hour, draw_minute=draw_minute)
    weather_by_date = {
        str(row["data_sorteio"]): row
        for _, row in hourly.iterrows()
    }
    refs = _temperature_reference_map(hourly)
    rows: List[Dict[str, object]] = []
    for _, contest in contests.iterrows():
        draw_date = str(contest["data_sorteio"])
        weather_row = weather_by_date.get(draw_date)
        status = "ok" if weather_row is not None else "sem_dado_horario"
        temp = _safe_float(weather_row.get("temperature_2m")) if weather_row is not None else float("nan")
        apparent = _safe_float(weather_row.get("apparent_temperature")) if weather_row is not None else float("nan")
        humidity = _safe_float(weather_row.get("relative_humidity_2m")) if weather_row is not None else float("nan")
        pressure = _safe_float(weather_row.get("surface_pressure")) if weather_row is not None else float("nan")
        precipitation = _safe_float(weather_row.get("precipitation")) if weather_row is not None else float("nan")
        ref = refs.get(draw_date, float("nan"))
        anomaly = round(float(temp - ref), 6) if not pd.isna(temp) and not pd.isna(ref) else float("nan")
        rows.append(
            {
                "concurso": int(contest["concurso"]),
                "data_sorteio": draw_date,
                "horario_brasilia_assumido": f"{int(draw_hour):02d}:{int(draw_minute):02d}",
                "cidade_sorteio": contest.get("cidade_sorteio", ""),
                "uf_sorteio": contest.get("uf_sorteio", ""),
                "cidade_sorteio_normalizada": normalize_context_text(contest.get("cidade_sorteio", "")),
                "uf_sorteio_normalizada": normalize_context_text(contest.get("uf_sorteio", "")),
                "latitude": geocode.get("latitude"),
                "longitude": geocode.get("longitude"),
                "clima_fonte": CLIMATE_SOURCE,
                "clima_status": status,
                "clima_temperature_2m": round(float(temp), 6) if not pd.isna(temp) else pd.NA,
                "clima_apparent_temperature": round(float(apparent), 6) if not pd.isna(apparent) else pd.NA,
                "clima_relative_humidity_2m": round(float(humidity), 6) if not pd.isna(humidity) else pd.NA,
                "clima_surface_pressure": round(float(pressure), 6) if not pd.isna(pressure) else pd.NA,
                "clima_precipitation": round(float(precipitation), 6) if not pd.isna(precipitation) else pd.NA,
                "clima_temperature_media_30d": round(float(ref), 6) if not pd.isna(ref) else pd.NA,
                "clima_temperature_anomalia": round(float(anomaly), 6) if not pd.isna(anomaly) else pd.NA,
                "clima_temperatura_faixa": _temperature_band(temp),
                "clima_sensacao_faixa": _temperature_band(apparent),
                "clima_umidade_faixa": _humidity_band(humidity),
                "clima_chuva_faixa": _precipitation_band(precipitation),
                "clima_anomalia_faixa": _anomaly_band(anomaly),
            }
        )
    return rows


def build_climate_features(
    concursos: pd.DataFrame,
    *,
    cache_dir: Path,
    draw_hour: int,
    draw_minute: int,
    timeout_seconds: float = 30.0,
    retries: int = 3,
    sleep_seconds: float = 0.05,
    max_locations: int = 0,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if concursos.empty:
        raise ValueError("Historico local nao encontrado.")
    df = concursos.copy().sort_values("concurso").reset_index(drop=True)
    df["cidade_sorteio_normalizada"] = df["cidade_sorteio"].map(normalize_context_text)
    df["uf_sorteio_normalizada"] = df["uf_sorteio"].map(normalize_context_text)
    grouped = (
        df[df["cidade_sorteio_normalizada"].astype(bool) & df["uf_sorteio_normalizada"].astype(bool)]
        .groupby(["cidade_sorteio_normalizada", "uf_sorteio_normalizada"], dropna=False)
        .size()
        .sort_values(ascending=False)
    )
    total_locations = int(len(grouped))
    if max_locations and int(max_locations) > 0:
        grouped = grouped.head(int(max_locations))

    all_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    for location_idx, ((city, uf), count) in enumerate(grouped.items(), start=1):
        location_df = df[(df["cidade_sorteio_normalizada"] == city) & (df["uf_sorteio_normalizada"] == uf)].copy()
        start_date = str(pd.to_datetime(location_df["data_sorteio"]).min().date())
        end_date = str(pd.to_datetime(location_df["data_sorteio"]).max().date())
        row_summary = {
            "cidade": city,
            "uf": uf,
            "concursos": int(count),
            "data_inicial": start_date,
            "data_final": end_date,
            "localidades_total": total_locations,
            "localidades_processadas_planejadas": int(len(grouped)),
            "status": "ok",
            "erro": "",
        }
        try:
            geocode = geocode_city(
                city=city,
                uf=uf,
                cache_dir=cache_dir,
                timeout_seconds=timeout_seconds,
                retries=retries,
                sleep_seconds=sleep_seconds,
                force=force,
            )
            weather = fetch_weather_range(
                latitude=float(geocode["latitude"]),
                longitude=float(geocode["longitude"]),
                start_date=start_date,
                end_date=end_date,
                cache_dir=cache_dir,
                timeout_seconds=timeout_seconds,
                retries=retries,
                sleep_seconds=sleep_seconds,
                force=force,
            )
            all_rows.extend(
                _build_rows_for_location(
                    location_df,
                    geocode=geocode,
                    weather=weather,
                    draw_hour=draw_hour,
                    draw_minute=draw_minute,
                )
            )
            row_summary["latitude"] = geocode.get("latitude")
            row_summary["longitude"] = geocode.get("longitude")
            row_summary["linhas_clima"] = int(len(location_df))
        except Exception as exc:  # pragma: no cover - integration resilience
            row_summary["status"] = "erro"
            row_summary["erro"] = str(exc)
            row_summary["linhas_clima"] = 0
            if logger:
                logger.warning("Falha climatica %s/%s: %s", city, uf, exc)
        summary_rows.append(row_summary)
        if logger:
            logger.info("Clima %s/%s processado (%s/%s)", city, uf, location_idx, len(grouped))

    climate = _finalize_climate_rows(all_rows)
    summary = pd.DataFrame(summary_rows)
    return climate, summary


def climate_lookup_for_target(
    concursos: pd.DataFrame,
    climate_features: pd.DataFrame,
    *,
    cache_dir: Path,
    draw_hour: int,
    draw_minute: int,
    timeout_seconds: float = 30.0,
    retries: int = 3,
    sleep_seconds: float = 0.05,
    force: bool = False,
) -> Dict[str, object]:
    target = build_target_context(concursos, draw_hour=draw_hour, draw_minute=draw_minute)
    city = target.cidade_sorteio_assumida
    uf = target.uf_sorteio_assumida
    if not city or not uf:
        return neutral_target_climate("localidade_indisponivel")
    try:
        geocode = geocode_city(
            city=city,
            uf=uf,
            cache_dir=cache_dir,
            timeout_seconds=timeout_seconds,
            retries=retries,
            sleep_seconds=sleep_seconds,
            force=force,
        )
        start = (pd.to_datetime(target.data_proximo_concurso) - pd.Timedelta(days=1)).date().isoformat()
        end = target.data_proximo_concurso
        weather = fetch_weather_range(
            latitude=float(geocode["latitude"]),
            longitude=float(geocode["longitude"]),
            start_date=start,
            end_date=end,
            cache_dir=cache_dir,
            timeout_seconds=timeout_seconds,
            retries=retries,
            sleep_seconds=sleep_seconds,
            force=force,
        )
        rows = _build_rows_for_location(
            pd.DataFrame(
                [
                    {
                        "concurso": int(target.concurso_alvo),
                        "data_sorteio": target.data_proximo_concurso,
                        "cidade_sorteio": city,
                        "uf_sorteio": uf,
                    }
                ]
            ),
            geocode=geocode,
            weather=weather,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
        )
        if not rows:
            return neutral_target_climate("sem_linha_climatica_alvo")
        row = rows[0]
        row["clima_temperature_media_30d"] = _target_temperature_reference(climate_features, city=city, uf=uf)
        temp = _safe_float(row.get("clima_temperature_2m"))
        ref = _safe_float(row.get("clima_temperature_media_30d"))
        anomaly = round(float(temp - ref), 6) if not pd.isna(temp) and not pd.isna(ref) else float("nan")
        row["clima_temperature_anomalia"] = round(float(anomaly), 6) if not pd.isna(anomaly) else pd.NA
        row["clima_anomalia_faixa"] = _anomaly_band(anomaly)
        finalized = _finalize_climate_rows([row])
        return dict(finalized.iloc[0]) if not finalized.empty else neutral_target_climate("sem_dado_finalizado")
    except Exception as exc:  # pragma: no cover - integration resilience
        return neutral_target_climate(f"erro_consulta_climatica:{exc}")


def _target_temperature_reference(climate_features: pd.DataFrame, *, city: str, uf: str) -> float:
    if climate_features.empty or "clima_temperature_2m" not in climate_features.columns:
        return float("nan")
    city_norm = normalize_context_text(city)
    uf_norm = normalize_context_text(uf)
    df = climate_features.copy()
    if "cidade_sorteio_normalizada" in df.columns and "uf_sorteio_normalizada" in df.columns:
        df = df[(df["cidade_sorteio_normalizada"] == city_norm) & (df["uf_sorteio_normalizada"] == uf_norm)]
    values = pd.to_numeric(df.get("clima_temperature_2m"), errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return round(float(values.tail(30).mean()), 6)


def neutral_target_climate(status: str = "neutro") -> Dict[str, object]:
    return {
        "clima_status": status,
        "clima_fonte": CLIMATE_SOURCE,
        "clima_temperature_2m": pd.NA,
        "clima_apparent_temperature": pd.NA,
        "clima_relative_humidity_2m": pd.NA,
        "clima_surface_pressure": pd.NA,
        "clima_precipitation": pd.NA,
        "clima_temperature_media_30d": pd.NA,
        "clima_temperature_anomalia": pd.NA,
        "clima_temperatura_faixa": "indisponivel",
        "clima_sensacao_faixa": "indisponivel",
        "clima_umidade_faixa": "indisponivel",
        "clima_pressao_faixa": "indisponivel",
        "clima_chuva_faixa": "indisponivel",
        "clima_anomalia_faixa": "indisponivel",
        "clima_assinatura": "indisponivel",
    }


def load_climate_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")
