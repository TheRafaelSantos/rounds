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
from .pipeline import LotofacilPipeline
from .supervised_calibration import load_supervised_calibration_status
from .top50_refinement_pipeline import Top50RefinementPipeline
from .top100_pipeline import Top100Pipeline


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
    .calibration-panel { border: 1px solid #d7dbdd; padding: 16px; border-radius: 6px; background: #ffffff; }
    .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin: 14px 0; }
    .status-card { border: 1px solid #e5e8e8; border-radius: 6px; padding: 10px; background: #fbfcfc; }
    .status-label { color: #566573; display: block; font-size: 12px; margin-bottom: 4px; }
    .status-value { font-size: 18px; font-weight: 700; }
    .table-scroll { overflow: auto; border: 1px solid #e5e8e8; border-radius: 6px; margin-top: 12px; }
    table { border-collapse: collapse; width: 100%; min-width: 780px; }
    th, td { border-bottom: 1px solid #edf1f2; padding: 8px; text-align: left; font-size: 13px; }
    th { background: #f4f6f7; color: #34495e; }
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
    <button onclick="updateBase()">Atualizar base incremental</button>
    <button onclick="climate()">Atualizar clima incremental</button>
    <button onclick="learning()">Aprendizado</button>
    <button onclick="top100()">Gerar 100 jogos</button>
    <button onclick="window.location='/top100-report'">Baixar Top 100</button>
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
    function statusCard(label, value) {
      return '<div class="status-card"><span class="status-label">' + escapeHtml(label) + '</span><span class="status-value">' + escapeHtml(value ?? '-') + '</span></div>';
    }
    function weightRows(rows) {
      if (!Array.isArray(rows) || rows.length === 0) {
        return '<div class="muted">Ainda não existe média de pesos vencedores. O arquivo será criado quando algum concurso for resolvido com 15 pontos.</div>';
      }
      return '<div class="metrics">' + rows.map(function(row) {
        return metricBar(row.componente, Number(row.peso_percentual || 0), true);
      }).join('') + '</div>';
    }
    function tableRows(rows, columns) {
      if (!Array.isArray(rows) || rows.length === 0) {
        return '<div class="muted">Sem registros ainda.</div>';
      }
      const header = columns.map(function(column) { return '<th>' + escapeHtml(column.label) + '</th>'; }).join('');
      const body = rows.map(function(row) {
        return '<tr>' + columns.map(function(column) {
          return '<td>' + escapeHtml(row[column.key] ?? '') + '</td>';
        }).join('') + '</tr>';
      }).join('');
      return '<div class="table-scroll"><table><thead><tr>' + header + '</tr></thead><tbody>' + body + '</tbody></table></div>';
    }
    function renderSupervised(data) {
      const state = data.state || {};
      const recent = data.recent_results || [];
      const best = data.best_results || [];
      const blocks = data.progress_blocks || [];
      const weights = data.weights || [];
      const engineWeights = data.engine_weights || {};
      const processed = state.total_contests_processed || recent.length || 0;
      const activeWeights = Object.keys(engineWeights).length
        ? Object.keys(engineWeights).map(function(key) {
            return {componente: key, peso_percentual: Number(engineWeights[key] || 0) * 100};
          })
        : weights;
      return '<section class="calibration-panel">' +
        '<div class="game-head"><h2>Aprendizado supervisionado</h2><span class="tag">' + escapeHtml(state.status || 'sem status') + '</span></div>' +
        '<div class="status-grid">' +
          statusCard('Concurso atual', state.current_concurso || '-') +
          statusCard('Último processado', state.last_concurso || '-') +
          statusCard('Concursos aprendidos', processed) +
          statusCard('Progresso elegível', state.progress_percent !== undefined ? Number(state.progress_percent).toFixed(2) + '%' : '-') +
          statusCard('Concursos elegíveis', state.eligible_target_count || '-') +
          statusCard('Pendentes elegíveis', state.remaining_eligible_count ?? '-') +
          statusCard('Próximo pendente', state.next_pending_concurso || '-') +
          statusCard('Primeiro elegível', state.first_eligible_concurso || '-') +
          statusCard('Pulados sem histórico', state.skipped_min_history_count ?? '-') +
          statusCard('Rank médio antes', state.rank_before_avg ?? '-') +
          statusCard('Rank médio depois', state.rank_after_avg ?? '-') +
          statusCard('Melhora média', state.rank_improvement_avg ?? '-') +
          statusCard('Melhor rank depois', state.best_rank_after ?? '-') +
          statusCard('Último rank antes', state.last_rank_antes ?? '-') +
          statusCard('Último rank depois', state.last_rank_depois ?? '-') +
          statusCard('Última melhora', state.last_melhora_rank ?? '-') +
          statusCard('Amostras/concurso', state.samples || '-') +
          statusCard('Histórico mínimo', state.min_history || '-') +
          statusCard('Tempo execução atual', state.elapsed_seconds_current_run ? Number(state.elapsed_seconds_current_run).toFixed(0) + 's' : '-') +
        '</div>' +
        '<div class="exclusives"><strong>Como ler:</strong> rank menor e percentil maior indicam que a sequência real ficou melhor posicionada com os pesos aprendidos. Concursos sem histórico mínimo são pulados para evitar olhar resultado futuro.</div>' +
        '<section class="comparison"><h2>Pesos atualmente aplicados no motor</h2>' + weightRows(activeWeights) + '</section>' +
        '<section class="comparison"><h2>Evolução por blocos de concursos</h2>' + tableRows(blocks, [
          {key: 'bloco', label: 'Bloco'},
          {key: 'concursos', label: 'Concursos'},
          {key: 'rank_antes_medio', label: 'Rank médio antes'},
          {key: 'rank_depois_medio', label: 'Rank médio depois'},
          {key: 'melhora_media', label: 'Melhora média'},
          {key: 'percentil_depois_medio', label: 'Percentil depois'}
        ]) + '</section>' +
        '<section class="comparison"><h2>Melhores posicionamentos aprendidos</h2>' + tableRows(best, [
          {key: 'concurso', label: 'Concurso'},
          {key: 'rank_antes', label: 'Rank antes'},
          {key: 'rank_depois', label: 'Rank depois'},
          {key: 'melhora_rank', label: 'Melhora'},
          {key: 'percentil_depois', label: 'Percentil depois'},
          {key: 'jogo_real', label: 'Jogo real'}
        ]) + '</section>' +
        '<section class="comparison"><h2>Últimos concursos aprendidos</h2>' + tableRows(recent, [
          {key: 'concurso', label: 'Concurso'},
          {key: 'rank_antes', label: 'Rank antes'},
          {key: 'rank_depois', label: 'Rank depois'},
          {key: 'melhora_rank', label: 'Melhora'},
          {key: 'percentil_antes', label: 'Percentil antes'},
          {key: 'percentil_depois', label: 'Percentil depois'},
          {key: 'jogo_real', label: 'Jogo real'}
        ]) + '</section>' +
      '</section>';
    }
    function renderTop100(data) {
      const rows = Array.isArray(data.games) ? data.games : [];
      if (!rows.length) {
        return '';
      }
      return '<section class="calibration-panel">' +
        '<div class="game-head"><h2>Ranking 100 jogos</h2><span class="tag">' + escapeHtml(data.metodo || 'top100') + '</span></div>' +
        '<div class="status-grid">' +
          statusCard('Concurso alvo', data.concurso_alvo || '-') +
          statusCard('Data concurso', data.data_proximo_concurso || '-') +
          statusCard('Jogos gerados', data.selected_rows || rows.length) +
          statusCard('Pool analisado', data.top_pool || '-') +
        '</div>' +
        '<section class="comparison"><h2>Top 10</h2>' + tableRows(rows.slice(0, 10), [
          {key: 'rank_top100', label: 'Rank'},
          {key: 'nums', label: 'Jogo'},
          {key: 'grupo_top', label: 'Grupo'},
          {key: 'primeira_dezena_top100', label: 'Primeira dezena'},
          {key: 'score_top100', label: 'Score Top 100'},
          {key: 'score_final', label: 'Score base'},
          {key: 'score_combinatorio_avancado', label: 'Comb. avançado'},
          {key: 'score_grafo_dezenas', label: 'Grafo'},
          {key: 'score_complemento_ausentes', label: 'Complemento'},
          {key: 'score_detector_falso_positivo', label: 'Anti falso positivo'}
        ]) + '</section>' +
        '<section class="comparison"><h2>Top 100 completo</h2>' + tableRows(rows.slice(0, 100), [
          {key: 'rank_top100', label: 'Rank'},
          {key: 'nums', label: 'Jogo'},
          {key: 'primeira_dezena_top100', label: 'Primeira dezena'},
          {key: 'score_top100', label: 'Score Top 100'},
          {key: 'score_top50_refinado', label: 'Score refinado'},
          {key: 'refinador_top50_aplicado', label: 'Refinador'},
          {key: 'criterio_top100', label: 'Critério'}
        ]) + '</section>' +
      '</section>';
    }
    function refinementWeightRows(rows, columnPrefix) {
      if (!Array.isArray(rows) || rows.length === 0) {
        return '<div class="muted">Sem pesos refinados ainda.</div>';
      }
      const key = columnPrefix === 'neg' ? 'peso_penalizador_percentual' : 'peso_positivo_percentual';
      const filtered = rows.filter(function(row) { return Number(row[key] || 0) > 0; })
        .sort(function(a, b) { return Number(b[key] || 0) - Number(a[key] || 0); })
        .slice(0, 12);
      if (!filtered.length) {
        return '<div class="muted">Sem pesos relevantes nesta categoria.</div>';
      }
      return '<div class="metrics">' + filtered.map(function(row) {
        return metricBar(row.feature, Number(row[key] || 0), columnPrefix !== 'neg');
      }).join('') + '</div>';
    }
    function renderTop50Refinement(data) {
      const state = data.state || {};
      const recent = data.recent_results || [];
      const best = data.best_results || [];
      const blocks = data.progress_blocks || [];
      const weights = data.weights || [];
      const processed = state.total_contests_processed || recent.length || 0;
      return '<section class="calibration-panel">' +
        '<div class="game-head"><h2>Motor 3.0 Refinador Top50</h2><span class="tag">' + escapeHtml(state.status || 'sem status') + '</span></div>' +
        '<div class="status-grid">' +
          statusCard('Concurso atual', state.current_concurso || '-') +
          statusCard('Último processado', state.last_concurso || '-') +
          statusCard('Concursos refinados', processed) +
          statusCard('Progresso elegível', state.progress_percent !== undefined ? Number(state.progress_percent).toFixed(2) + '%' : '-') +
          statusCard('Pendentes elegíveis', state.remaining_eligible_count ?? '-') +
          statusCard('Próximo pendente', state.next_pending_concurso || '-') +
          statusCard('Rank médio antes', state.rank_before_avg ?? '-') +
          statusCard('Rank médio refinado', state.rank_after_avg ?? '-') +
          statusCard('Melhora média', state.rank_improvement_avg ?? '-') +
          statusCard('Hit@50 antes', state.hit_top50_before !== undefined ? Number(state.hit_top50_before).toFixed(2) + '%' : '-') +
          statusCard('Hit@50 refinado', state.hit_top50_after !== undefined ? Number(state.hit_top50_after).toFixed(2) + '%' : '-') +
          statusCard('Hit@100 antes', state.hit_top100_before !== undefined ? Number(state.hit_top100_before).toFixed(2) + '%' : '-') +
          statusCard('Hit@100 refinado', state.hit_top100_after !== undefined ? Number(state.hit_top100_after).toFixed(2) + '%' : '-') +
          statusCard('Último rank antes', state.last_rank_top100_antes ?? '-') +
          statusCard('Último rank refinado', state.last_rank_top50_refinado ?? '-') +
          statusCard('Última melhora', state.last_melhora_rank_refinador ?? '-') +
          statusCard('Pool', state.top_pool || '-') +
          statusCard('Histórico mínimo', state.min_history || '-') +
          statusCard('Tempo execução atual', state.elapsed_seconds_current_run ? Number(state.elapsed_seconds_current_run).toFixed(0) + 's' : '-') +
        '</div>' +
        '<div class="exclusives"><strong>Como ler:</strong> este painel mostra aprendizado pós-erro em concursos já encerrados. Ele aprende quais sinais subiriam o gabarito histórico acima dos falsos positivos e salva pesos para concursos futuros.</div>' +
        '<section class="comparison"><h2>Pesos que puxam para cima</h2>' + refinementWeightRows(weights, 'pos') + '</section>' +
        '<section class="comparison"><h2>Pesos penalizadores de falso Top50</h2>' + refinementWeightRows(weights, 'neg') + '</section>' +
        '<section class="comparison"><h2>Evolução por blocos</h2>' + tableRows(blocks, [
          {key: 'bloco', label: 'Bloco'},
          {key: 'concursos', label: 'Concursos'},
          {key: 'rank_antes_medio', label: 'Rank antes'},
          {key: 'rank_refinado_medio', label: 'Rank refinado'},
          {key: 'melhora_media', label: 'Melhora'},
          {key: 'hit_top50_refinado_pct', label: 'Hit@50 refinado'},
          {key: 'hit_top100_refinado_pct', label: 'Hit@100 refinado'}
        ]) + '</section>' +
        '<section class="comparison"><h2>Melhores refinamentos</h2>' + tableRows(best, [
          {key: 'concurso', label: 'Concurso'},
          {key: 'rank_top100_antes', label: 'Rank antes'},
          {key: 'rank_top50_refinado', label: 'Rank refinado'},
          {key: 'melhora_rank_refinador', label: 'Melhora'},
          {key: 'jogo_real', label: 'Jogo real'}
        ]) + '</section>' +
        '<section class="comparison"><h2>Últimos concursos refinados</h2>' + tableRows(recent, [
          {key: 'concurso', label: 'Concurso'},
          {key: 'rank_top100_antes', label: 'Rank antes'},
          {key: 'rank_top50_refinado', label: 'Rank refinado'},
          {key: 'hit_top50_refinado', label: 'Top50'},
          {key: 'hit_top100_refinado', label: 'Top100'},
          {key: 'jogo_real', label: 'Jogo real'}
        ]) + '</section>' +
      '</section>';
    }
    function renderLearning(data) {
      const supervised = data.supervised || {};
      const top50 = data.top50 || {};
      const supervisedState = supervised.state || {};
      const top50State = top50.state || {};
      return '<section class="calibration-panel">' +
        '<div class="game-head"><h2>Aprendizado unificado</h2><span class="tag">motor 3.0</span></div>' +
        '<div class="status-grid">' +
          statusCard('Supervisionado', supervisedState.status || '-') +
          statusCard('Supervisionado progresso', supervisedState.progress_percent !== undefined ? Number(supervisedState.progress_percent).toFixed(2) + '%' : '-') +
          statusCard('Supervisionado rank depois', supervisedState.rank_after_avg ?? '-') +
          statusCard('Top50', top50State.status || '-') +
          statusCard('Top50 progresso', top50State.progress_percent !== undefined ? Number(top50State.progress_percent).toFixed(2) + '%' : '-') +
          statusCard('Top50 rank refinado', top50State.rank_after_avg ?? '-') +
          statusCard('Top50 Hit@50', top50State.hit_top50_after !== undefined ? Number(top50State.hit_top50_after).toFixed(2) + '%' : '-') +
          statusCard('Top50 Hit@100', top50State.hit_top100_after !== undefined ? Number(top50State.hit_top100_after).toFixed(2) + '%' : '-') +
        '</div>' +
        '<div class="exclusives"><strong>Como ler:</strong> este botão centraliza todos os aprendizados. Os serviços 24/7 continuam treinando por trás; aqui você acompanha o estado consolidado sem precisar escolher entre painéis separados.</div>' +
      '</section>' +
      renderSupervised(supervised) +
      renderTop50Refinement(top50);
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
    async function climate() { await request('/api/climate', {method: 'POST'}); }
    async function learning() {
      const data = await request('/api/learning/status');
      document.getElementById('games').innerHTML = renderLearning(data);
    }
    async function top100() {
      const data = await request('/api/top100', {method: 'POST'});
      document.getElementById('games').innerHTML = renderTop100(data);
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
            if self.path == "/api/learning/status":
                self._handle_json(
                    lambda: {
                        "supervised": load_supervised_calibration_status(
                            state_json_path=config.supervised_calibration_state_json_path,
                            results_csv_path=config.supervised_calibration_results_csv_path,
                            summary_csv_path=config.supervised_calibration_summary_csv_path,
                            weights_csv_path=config.supervised_calibration_weights_csv_path,
                            weights_json_path=config.supervised_calibration_weights_json_path,
                        ),
                        "top50": Top50RefinementPipeline(config=config, logger=logger).status(),
                    }
                )
                return
            if self.path == "/top100-report":
                report_path = config.top100_prediction_report_path
                if not report_path.exists():
                    _json_response(self, HTTPStatus.NOT_FOUND, {"error": "relatorio ainda nao gerado; use Gerar Top 100 primeiro"})
                    return
                body = report_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/markdown; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="lotofacil_prediction_top100_report.md"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "rota nao encontrada"})

        def do_POST(self) -> None:
            if self.path == "/api/update":
                self._handle_json(lambda: {"message": LotofacilPipeline(config=config, logger=logger).update(force_full=False).to_console()})
                return
            if self.path == "/api/climate":
                self._handle_json(lambda: ClimatePipeline(config=config, logger=logger).run(draw_hour=20, draw_minute=0).__dict__)
                return
            if self.path == "/api/top100":
                def top100_payload() -> dict:
                    summary = Top100Pipeline(config=config, logger=logger).predict(
                        top_count=100,
                        top_pool=10000,
                        max_overlap=11,
                        draw_hour=20,
                        draw_minute=0,
                        exhaustive_limit=None,
                    )
                    payload = summary.__dict__.copy()
                    payload["games"] = _read_csv_records(config.top100_prediction_csv_path)
                    return payload

                self._handle_json(top100_payload)
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
