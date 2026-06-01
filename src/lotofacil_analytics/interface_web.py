from __future__ import annotations

import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

from .config import AppConfig
from .decision_layer_pipeline import DecisionLayerPipeline
from .pipeline import LotofacilPipeline
from .predictor_pipeline import PredictorPipeline


def _html_page() -> str:
    return """<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Lotofacil Analytics</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 32px; max-width: 980px; color: #17202a; }
    header { border-bottom: 1px solid #d7dbdd; margin-bottom: 24px; padding-bottom: 12px; }
    button { margin: 4px 8px 4px 0; padding: 10px 14px; border: 1px solid #85929e; background: #f8f9f9; cursor: pointer; }
    button:hover { background: #eaecee; }
    .games { display: grid; gap: 12px; margin: 18px 0; }
    .game { border: 1px solid #d7dbdd; padding: 14px; border-radius: 6px; font-size: 18px; }
    pre { background: #f4f6f7; padding: 12px; overflow: auto; border: 1px solid #d7dbdd; }
    .muted { color: #566573; }
  </style>
</head>
<body>
  <header>
    <h1>Lotofacil Analytics</h1>
    <p class="muted">Execucao local em Python. Sugestoes estatisticas, sem garantia de acerto.</p>
  </header>
  <button onclick="status()">Status</button>
  <button onclick="updateBase()">Atualizar base</button>
  <button onclick="predictSingle()">Gerar jogo unico</button>
  <button onclick="predict()">Gerar 2 jogos</button>
  <button onclick="window.location='/report'">Baixar relatorio</button>
  <button onclick="window.location='/single-report'">Baixar relatorio jogo unico</button>
  <section class="games" id="games"></section>
  <pre id="output">Pronto.</pre>
  <script>
    async function request(path, options) {
      const output = document.getElementById('output');
      output.textContent = 'Processando...';
      const response = await fetch(path, options || {});
      const data = await response.json();
      output.textContent = JSON.stringify(data, null, 2);
      return data;
    }
    async function status() { await request('/api/status'); }
    async function updateBase() { await request('/api/update', {method: 'POST'}); }
    async function predict() {
      const data = await request('/api/predict', {method: 'POST'});
      const games = document.getElementById('games');
      games.innerHTML = '';
      if (data.jogo_1) {
        games.innerHTML += '<div class="game"><strong>Jogo 1</strong><br>' + data.jogo_1 + '</div>';
        games.innerHTML += '<div class="game"><strong>Jogo 2</strong><br>' + data.jogo_2 + '</div>';
      }
    }
    async function predictSingle() {
      const data = await request('/api/predict-single', {method: 'POST'});
      const games = document.getElementById('games');
      games.innerHTML = '';
      if (data.jogo_unico) {
        games.innerHTML += '<div class="game"><strong>Jogo unico</strong><br>' + data.jogo_unico + '</div>';
      }
    }
  </script>
</body>
</html>
"""


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(config: AppConfig, logger: logging.Logger) -> type[BaseHTTPRequestHandler]:
    class LotofacilHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            logger.info("web | " + fmt, *args)

        def do_GET(self) -> None:
            if self.path == "/":
                body = _html_page().encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/api/status":
                self._handle_json(lambda: LotofacilPipeline(config=config, logger=logger).status().__dict__)
                return
            if self.path == "/report":
                report_path = config.prediction_report_path
                if not report_path.exists():
                    _json_response(self, HTTPStatus.NOT_FOUND, {"error": "relatorio ainda nao gerado; use Gerar 2 jogos primeiro"})
                    return
                body = report_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/markdown; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="lotofacil_prediction_report.md"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/single-report":
                report_path = config.single_prediction_report_path
                if not report_path.exists():
                    _json_response(self, HTTPStatus.NOT_FOUND, {"error": "relatorio ainda nao gerado; use Gerar jogo unico primeiro"})
                    return
                body = report_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/markdown; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="lotofacil_prediction_single_report.md"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "rota nao encontrada"})

        def do_POST(self) -> None:
            if self.path == "/api/update":
                self._handle_json(lambda: {"message": LotofacilPipeline(config=config, logger=logger).update(force_full=False).to_console()})
                return
            if self.path == "/api/predict":
                self._handle_json(lambda: PredictorPipeline(config=config, logger=logger).predict(
                    seed=123,
                    candidate_pool=10000,
                    top_games=100,
                    generations=20,
                    population=80,
                    max_overlap=8,
                    engine="exaustivo",
                ).__dict__)
                return
            if self.path == "/api/predict-single":
                self._handle_json(lambda: DecisionLayerPipeline(config=config, logger=logger).predict_single(
                    seed=123,
                    candidate_pool=10000,
                    top_games=100,
                    generations=20,
                    population=80,
                    draw_hour=20,
                    draw_minute=0,
                    engine="exaustivo",
                    exhaustive_limit=None,
                    weight_profile="padrao_atual",
                ).__dict__)
                return
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "rota nao encontrada"})

        def _handle_json(self, fn: Callable[[], dict]) -> None:
            try:
                payload = fn()
                serializable = {key: str(value) for key, value in payload.items()}
                _json_response(self, HTTPStatus.OK, serializable)
            except Exception as exc:
                logger.exception("Erro na interface web: %s", exc)
                _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    return LotofacilHandler


def run_web_server(config: AppConfig, logger: logging.Logger, *, host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, int(port)), make_handler(config, logger))
    logger.info("Servidor web local em http://%s:%s", host, port)
    print(f"Servidor web local: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Servidor encerrado.")
    finally:
        server.server_close()
