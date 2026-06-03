from __future__ import annotations

import json
import logging
import math
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .climate_pipeline import ClimatePipeline
from .config import AppConfig
from .decision_layer_pipeline import DecisionLayerPipeline
from .mandel_pipeline import MandelPipeline
from .pipeline import LotofacilPipeline
from .predictor_pipeline import PredictorPipeline
from .transition_pipeline import TransitionPipeline


def _html_page() -> str:
    return """<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Lotofacil Analytics</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 32px; max-width: 1220px; color: #17202a; }
    header { border-bottom: 1px solid #d7dbdd; margin-bottom: 24px; padding-bottom: 12px; }
    h1 { margin-bottom: 8px; }
    button { margin: 4px 8px 4px 0; padding: 10px 14px; border: 1px solid #85929e; background: #f8f9f9; cursor: pointer; }
    button:hover { background: #eaecee; }
    .toolbar { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 24px; }
    .toolbar button { margin: 0; }
    .games { display: grid; gap: 16px; margin: 18px 0; }
    .game-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }
    .game { border: 1px solid #d7dbdd; padding: 16px; border-radius: 6px; background: #ffffff; }
    .game-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 10px; }
    .game h2, .comparison h2 { font-size: 20px; margin: 0; }
    .numbers { font-size: 22px; line-height: 1.35; margin: 10px 0 14px; word-spacing: 3px; }
    .tag { display: inline-block; border: 1px solid #aeb6bf; border-radius: 999px; padding: 3px 8px; font-size: 12px; color: #34495e; background: #f8f9f9; white-space: nowrap; }
    .meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px; margin: 12px 0; }
    .meta-item { border: 1px solid #e5e8e8; border-radius: 6px; padding: 8px; background: #fbfcfc; }
    .meta-label { display: block; color: #566573; font-size: 12px; margin-bottom: 2px; }
    .meta-value { font-size: 15px; font-weight: 700; }
    .metrics { display: grid; gap: 9px; margin-top: 12px; }
    .metric-row { display: grid; grid-template-columns: minmax(130px, 170px) 1fr 58px; gap: 10px; align-items: center; }
    .metric-label { color: #34495e; font-size: 13px; }
    .metric-value { text-align: right; font-variant-numeric: tabular-nums; font-size: 13px; }
    .bar { height: 9px; border-radius: 999px; background: #e5e8e8; overflow: hidden; }
    .bar-fill { height: 100%; background: #1f618d; }
    .bar-fill.alt { background: #117864; }
    .exclusives { margin-top: 12px; padding-top: 12px; border-top: 1px solid #edf1f2; color: #34495e; font-size: 14px; }
    .comparison { border: 1px solid #d7dbdd; padding: 16px; border-radius: 6px; background: #fbfcfc; }
    .compare-grid { display: grid; gap: 10px; margin-top: 14px; }
    .compare-row { display: grid; grid-template-columns: 150px 1fr 1fr; gap: 12px; align-items: center; }
    .compare-title { color: #34495e; font-size: 13px; }
    .compare-cell { display: grid; grid-template-columns: 1fr 56px; gap: 8px; align-items: center; }
    .compare-empty { color: #85929e; font-size: 13px; }
    pre { background: #f4f6f7; padding: 12px; overflow: auto; border: 1px solid #d7dbdd; }
    .muted { color: #566573; }
    @media (max-width: 700px) {
      body { margin: 18px; }
      .metric-row, .compare-row { grid-template-columns: 1fr; }
      .metric-value { text-align: left; }
      .numbers { font-size: 19px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Lotofacil Analytics</h1>
    <p class="muted">Execucao local em Python. Sugestoes estatisticas, sem garantia de acerto.</p>
  </header>
  <div class="toolbar">
    <button onclick="status()">Status</button>
    <button onclick="updateBase()">Atualizar base</button>
    <button onclick="transitions()">Analisar transições</button>
    <button onclick="climate()">Atualizar clima</button>
    <button onclick="predictSingle()">Gerar jogo unico</button>
    <button onclick="predict()">Gerar 2 jogos</button>
    <button onclick="mandel()">Plano Mandel/bolão</button>
    <button onclick="window.location='/report'">Baixar relatorio</button>
    <button onclick="window.location='/single-report'">Baixar relatorio jogo unico</button>
    <button onclick="window.location='/mandel-report'">Baixar relatorio Mandel</button>
  </div>
  <section class="games" id="games"></section>
  <pre id="output">Pronto.</pre>
  <script>
    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"']/g, function(char) {
        return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'}[char];
      });
    }
    function scoreValue(value) {
      const number = Number(value);
      return Number.isFinite(number) ? number : null;
    }
    function metricBar(label, value, alt) {
      const number = scoreValue(value);
      if (number === null) {
        return '';
      }
      const pct = Math.max(0, Math.min(100, number));
      const fillClass = alt ? 'bar-fill alt' : 'bar-fill';
      return '<div class="metric-row">' +
        '<div class="metric-label">' + escapeHtml(label) + '</div>' +
        '<div class="bar"><div class="' + fillClass + '" style="width:' + pct.toFixed(2) + '%"></div></div>' +
        '<div class="metric-value">' + number.toFixed(2) + '</div>' +
      '</div>';
    }
    function metaItem(label, value) {
      if (value === undefined || value === null || value === '') {
        return '';
      }
      return '<div class="meta-item"><span class="meta-label">' + escapeHtml(label) + '</span><span class="meta-value">' + escapeHtml(value) + '</span></div>';
    }
    function renderGameCard(title, row, fallbackNums, isSecond) {
      const data = row || {};
      const nums = data.nums || fallbackNums || '';
      const criterio = data.criterio_selecao ? '<span class="tag">' + escapeHtml(data.criterio_selecao) + '</span>' : '';
      const metrics = [
        metricBar('Score final', data.score_final, false),
        metricBar('Decisão protegida', data.score_decisao_protegida, true),
        metricBar('Score contextual', data.score_contextual, false),
        metricBar('Score climático', data.score_climatico, false),
        metricBar('Temporal profundo', data.score_temporal_profundo, false),
        metricBar('Contexto protegido', data.score_contexto_protegido, true),
        metricBar('Score transição', data.score_transicao, false),
        metricBar('Anti-falso-negativo', data.score_cobertura_risco_falso_negativo, true),
        isSecond ? metricBar('Score portfólio', data.score_portfolio_jogo_2, true) : '',
        isSecond ? metricBar('Diversidade vs Jogo 1', data.score_diversidade_jogo_2, true) : '',
        isSecond ? metricBar('Força dos componentes', data.score_forca_componentes_jogo_2, true) : ''
      ].join('');
      const exclusives = isSecond && data.dezenas_exclusivas_jogo_2
        ? '<div class="exclusives"><strong>Dezenas exclusivas do Jogo 2:</strong> ' + escapeHtml(data.dezenas_exclusivas_jogo_2) + '</div>'
        : '';
      const risk = data.dezenas_risco_falso_negativo
        ? '<div class="exclusives"><strong>Dezenas de risco protegidas:</strong> ' + escapeHtml(data.dezenas_risco_falso_negativo) + '</div>'
        : '';
      return '<article class="game">' +
        '<div class="game-head"><h2>' + escapeHtml(title) + '</h2>' + criterio + '</div>' +
        '<div class="numbers">' + escapeHtml(nums) + '</div>' +
        '<div class="meta">' +
          metaItem('Rank', data.rank) +
          metaItem('Overlap Jogo 1', data.overlap_com_jogo_1) +
          metaItem('Únicas vs Jogo 1', data.dezenas_unicas_vs_jogo_1) +
          metaItem('Soma', data.soma) +
          metaItem('Pares', data.qtd_pares) +
          metaItem('Repetidas último', data.overlap_ultimo) +
          metaItem('Risco falso negativo', data.qtd_dezenas_risco_falso_negativo) +
        '</div>' +
        '<div class="metrics">' + metrics + '</div>' +
        risk +
        exclusives +
      '</article>';
    }
    function compareCell(value) {
      const number = scoreValue(value);
      if (number === null) {
        return '<div class="compare-empty">-</div>';
      }
      const pct = Math.max(0, Math.min(100, number));
      return '<div class="compare-cell"><div class="bar"><div class="bar-fill" style="width:' + pct.toFixed(2) + '%"></div></div><div class="metric-value">' + number.toFixed(2) + '</div></div>';
    }
    function compareRow(label, first, second) {
      return '<div class="compare-row">' +
        '<div class="compare-title">' + escapeHtml(label) + '</div>' +
        compareCell(first) +
        compareCell(second) +
      '</div>';
    }
    function renderComparison(rows) {
      if (!Array.isArray(rows) || rows.length < 2) {
        return '';
      }
      const first = rows[0] || {};
      const second = rows[1] || {};
      return '<section class="comparison">' +
        '<h2>Comparação visual dos scores</h2>' +
        '<div class="compare-grid">' +
          compareRow('Score final', first.score_final, second.score_final) +
          compareRow('Decisão protegida', first.score_decisao_protegida, second.score_decisao_protegida) +
          compareRow('Score contextual', first.score_contextual, second.score_contextual) +
          compareRow('Score climático', first.score_climatico, second.score_climatico) +
          compareRow('Temporal profundo', first.score_temporal_profundo, second.score_temporal_profundo) +
          compareRow('Contexto protegido', first.score_contexto_protegido, second.score_contexto_protegido) +
          compareRow('Score transição', first.score_transicao, second.score_transicao) +
          compareRow('Anti-falso-negativo', first.score_cobertura_risco_falso_negativo, second.score_cobertura_risco_falso_negativo) +
          compareRow('Score portfólio', null, second.score_portfolio_jogo_2) +
        '</div>' +
      '</section>';
    }
    function renderMandel(data) {
      const plan = Array.isArray(data.plan) ? data.plan : [];
      const preview = Array.isArray(data.games_preview) ? data.games_preview : [];
      const rows = plan.map(function(item) {
        return '<div class="compare-row">' +
          '<div class="compare-title">' + escapeHtml(item.tamanho_universo) + ' dezenas</div>' +
          '<div>' + escapeHtml(item.jogos_desdobramento_completo) + ' jogos</div>' +
          '<div>R$ ' + Number(item.custo_desdobramento_completo || 0).toLocaleString('pt-BR', {minimumFractionDigits: 2}) + '</div>' +
        '</div>';
      }).join('');
      const previewRows = preview.map(function(item) {
        return '<div class="exclusives"><strong>Jogo ' + escapeHtml(item.jogo) + ':</strong> ' + escapeHtml(item.nums) + '</div>';
      }).join('');
      return '<article class="game">' +
        '<div class="game-head"><h2>Plano Mandel/bolão</h2><span class="tag">desdobramento</span></div>' +
        '<div class="numbers">' + escapeHtml(data.universo_recomendado || '') + '</div>' +
        '<div class="meta">' +
          metaItem('Concurso', data.concurso_alvo) +
          metaItem('Universo', data.tamanho_universo) +
          metaItem('Completo', data.jogos_desdobramento_completo + ' jogos') +
          metaItem('Custo completo', 'R$ ' + Number(data.custo_desdobramento_completo || 0).toLocaleString('pt-BR', {minimumFractionDigits: 2})) +
          metaItem('Reduzido', data.jogos_fechamento_reduzido + ' jogos') +
          metaItem('Custo reduzido', 'R$ ' + Number(data.custo_fechamento_reduzido || 0).toLocaleString('pt-BR', {minimumFractionDigits: 2})) +
        '</div>' +
        '<div class="exclusives"><strong>Garantia:</strong> ' + escapeHtml(data.garantia_fechamento_reduzido || '') + '</div>' +
      '</article>' +
      '<section class="comparison"><h2>Custos por universo</h2><div class="compare-grid">' + rows + '</div></section>' +
      '<section class="comparison"><h2>Primeiros jogos do fechamento</h2>' + previewRows + '</section>';
    }
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
    async function transitions() { await request('/api/transitions', {method: 'POST'}); }
    async function climate() { await request('/api/climate', {method: 'POST'}); }
    async function mandel() {
      const data = await request('/api/mandel', {method: 'POST'});
      const games = document.getElementById('games');
      if (data.universo_recomendado) {
        games.innerHTML = renderMandel(data);
      } else {
        games.innerHTML = '';
      }
    }
    async function predict() {
      const data = await request('/api/predict', {method: 'POST'});
      const games = document.getElementById('games');
      const rows = Array.isArray(data.games) ? data.games : [];
      if (data.jogo_1) {
        const row1 = rows[0] || {nums: data.jogo_1};
        const row2 = rows[1] || {nums: data.jogo_2};
        games.innerHTML = '<div class="game-grid">' +
          renderGameCard('Jogo 1', row1, data.jogo_1, false) +
          renderGameCard('Jogo 2', row2, data.jogo_2, true) +
        '</div>' + renderComparison([row1, row2]);
      } else {
        games.innerHTML = '';
      }
    }
    async function predictSingle() {
      const data = await request('/api/predict-single', {method: 'POST'});
      const games = document.getElementById('games');
      const row = data.game || {};
      if (data.jogo_unico) {
        games.innerHTML = '<div class="game-grid">' + renderGameCard('Jogo unico', row, data.jogo_unico, false) + '</div>';
      } else {
        games.innerHTML = '';
      }
    }
  </script>
</body>
</html>
"""


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _read_csv_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    clean = df.astype(object).where(pd.notna(df), None)
    return clean.to_dict(orient="records")


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
            if self.path == "/mandel-report":
                report_path = config.mandel_report_path
                if not report_path.exists():
                    _json_response(self, HTTPStatus.NOT_FOUND, {"error": "relatorio ainda nao gerado; use Plano Mandel/bolao primeiro"})
                    return
                body = report_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/markdown; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="lotofacil_mandel_report.md"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "rota nao encontrada"})

        def do_POST(self) -> None:
            if self.path == "/api/update":
                self._handle_json(lambda: {"message": LotofacilPipeline(config=config, logger=logger).update(force_full=False).to_console()})
                return
            if self.path == "/api/transitions":
                self._handle_json(lambda: TransitionPipeline(config=config, logger=logger).run().__dict__)
                return
            if self.path == "/api/climate":
                self._handle_json(lambda: ClimatePipeline(config=config, logger=logger).run(draw_hour=20, draw_minute=0).__dict__)
                return
            if self.path == "/api/mandel":
                def mandel_payload() -> dict:
                    summary = MandelPipeline(config=config, logger=logger).run(
                        universe_size=18,
                        guarantee_hits=14,
                        max_reduced_games=80,
                        draw_hour=20,
                        draw_minute=0,
                    )
                    payload = summary.__dict__.copy()
                    payload["plan"] = _read_csv_records(config.mandel_plan_csv_path)
                    payload["games_preview"] = _read_csv_records(config.mandel_games_csv_path)[:20]
                    return payload

                self._handle_json(mandel_payload)
                return
            if self.path == "/api/predict":
                def predict_payload() -> dict:
                    summary = PredictorPipeline(config=config, logger=logger).predict(
                        seed=123,
                        candidate_pool=10000,
                        top_games=100,
                        generations=20,
                        population=80,
                        max_overlap=8,
                        engine="exaustivo",
                    )
                    payload = summary.__dict__.copy()
                    payload["games"] = _read_csv_records(config.prediction_csv_path)
                    return payload

                self._handle_json(predict_payload)
                return
            if self.path == "/api/predict-single":
                def predict_single_payload() -> dict:
                    summary = DecisionLayerPipeline(config=config, logger=logger).predict_single(
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
                    )
                    payload = summary.__dict__.copy()
                    rows = _read_csv_records(config.single_prediction_csv_path)
                    payload["game"] = rows[0] if rows else {}
                    return payload

                self._handle_json(predict_single_payload)
                return
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "rota nao encontrada"})

        def _handle_json(self, fn: Callable[[], dict]) -> None:
            try:
                payload = fn()
                serializable = _json_safe(payload)
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
