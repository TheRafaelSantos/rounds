from __future__ import annotations

import logging
import time
from typing import Any, Dict

import requests


class CaixaClientError(RuntimeError):
    pass


class CaixaLotofacilClient:
    BASE_URL = "https://servicebus2.caixa.gov.br/portaldeloterias/api/lotofacil"

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
                "Accept": "application/json",
                "User-Agent": "LotofacilAnalytics/0.1 (+https://github.com/TheRafaelSantos/rounds)",
            }
        )

    def fetch_latest(self) -> Dict[str, Any]:
        return self._get_json(self.BASE_URL)

    def fetch_contest(self, concurso: int) -> Dict[str, Any]:
        if concurso < 1:
            raise ValueError("concurso deve ser maior ou igual a 1.")
        return self._get_json(f"{self.BASE_URL}/{int(concurso)}")

    def _get_json(self, url: str) -> Dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, timeout=self.timeout_seconds)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise CaixaClientError(f"Resposta inesperada da CAIXA: {type(payload)!r}")
                if self.request_sleep_seconds:
                    time.sleep(self.request_sleep_seconds)
                return payload
            except (requests.RequestException, ValueError, CaixaClientError) as exc:
                last_error = exc
                wait = min(2.0 * attempt, 10.0)
                self.logger.warning("Falha ao consultar %s | tentativa %s/%s | %s", url, attempt, self.max_retries, exc)
                if attempt < self.max_retries:
                    time.sleep(wait)
        raise CaixaClientError(f"Falha ao consultar a API da CAIXA: {url} | ultimo erro: {last_error}")
