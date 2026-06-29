from __future__ import annotations

import logging
import time
from typing import Any, Dict

import requests


class CaixaClientError(RuntimeError):
    pass


class CaixaLotofacilClient:
    BASE_URL = "https://servicebus2.caixa.gov.br/portaldeloterias/api/lotofacil"
    FALLBACK_BASE_URL = "https://api.guidi.dev.br/loteria/lotofacil"

    def __init__(
        self,
        *,
        timeout_seconds: float,
        max_retries: int,
        request_sleep_seconds: float,
        logger: logging.Logger,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(1, int(max_retries))
        self.request_sleep_seconds = max(0.0, float(request_sleep_seconds))
        self.logger = logger
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
                "Referer": "https://loterias.caixa.gov.br/Paginas/Lotofacil.aspx",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0.0.0 Safari/537.36"
                ),
            }
        )

    def fetch_latest(self) -> Dict[str, Any]:
        return self._get_json(
            [
                ("caixa", self.BASE_URL),
                ("guidi", f"{self.FALLBACK_BASE_URL}/ultimo"),
            ]
        )

    def fetch_contest(self, concurso: int) -> Dict[str, Any]:
        if concurso < 1:
            raise ValueError("concurso deve ser maior ou igual a 1.")
        return self._get_json(
            [
                ("caixa", f"{self.BASE_URL}/{int(concurso)}"),
                ("guidi", f"{self.FALLBACK_BASE_URL}/{int(concurso)}"),
            ]
        )

    def _get_json(self, sources: list[tuple[str, str]]) -> Dict[str, Any]:
        last_error: Exception | None = None
        for source_name, url in sources:
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = self.session.get(url, timeout=self.timeout_seconds)
                    response.raise_for_status()
                    payload = response.json()
                    if not isinstance(payload, dict):
                        raise CaixaClientError(f"Resposta inesperada da fonte {source_name}: {type(payload)!r}")
                    if source_name != "caixa":
                        self.logger.warning("API da CAIXA indisponivel/bloqueada; usando fallback %s: %s", source_name, url)
                    if self.request_sleep_seconds:
                        time.sleep(self.request_sleep_seconds)
                    return payload
                except requests.HTTPError as exc:
                    last_error = exc
                    status_code = exc.response.status_code if exc.response is not None else None
                    self.logger.warning(
                        "Falha ao consultar %s (%s) | tentativa %s/%s | %s",
                        source_name,
                        url,
                        attempt,
                        self.max_retries,
                        exc,
                    )
                    if status_code in {401, 403, 404}:
                        break
                    if attempt < self.max_retries:
                        time.sleep(min(2.0 * attempt, 10.0))
                except (requests.RequestException, ValueError, CaixaClientError) as exc:
                    last_error = exc
                    wait = min(2.0 * attempt, 10.0)
                    self.logger.warning(
                        "Falha ao consultar %s (%s) | tentativa %s/%s | %s",
                        source_name,
                        url,
                        attempt,
                        self.max_retries,
                        exc,
                    )
                    if attempt < self.max_retries:
                        time.sleep(wait)
        urls = " | ".join(url for _source, url in sources)
        raise CaixaClientError(f"Falha ao consultar as APIs da Lotofacil: {urls} | ultimo erro: {last_error}")
