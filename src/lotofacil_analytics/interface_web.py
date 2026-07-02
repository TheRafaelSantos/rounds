from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Callable

import pandas as pd

from .climate_pipeline import ClimatePipeline
from .config import AppConfig
from .learning_pipeline import UnifiedLearningPipeline
from .pipeline import LotofacilPipeline
from .top100_pipeline import Top100Pipeline


_TOP100_JOB_LOCK = Lock()
_TOP100_JOB: dict[str, Any] = {
    "status": "idle",
    "started_at": None,
    "finished_at": None,
    "error": None,
    "summary": None,
}
_WEB_TOP100_TOP_COUNT = 100
_WEB_TOP100_TOP_POOL_DEFAULT = 500
_WEB_TOP100_EXHAUSTIVE_LIMIT_DEFAULT = 3000
_WEB_TOP100_MAX_OVERLAP_DEFAULT = 11
_LEARNING_JOB_LOCK = Lock()
_LEARNING_JOB: dict[str, Any] = {
    "status": "idle",
    "started_at": None,
    "finished_at": None,
    "error": None,
    "summary": None,
}


def _env_int(name: str, default: int, *, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        value = int(str(raw).strip())
    except ValueError:
        return int(default)
    return max(int(minimum), int(value))


def _top100_web_settings() -> dict[str, int]:
    top_pool = _env_int("LOTOFACIL_WEB_TOP100_POOL", _WEB_TOP100_TOP_POOL_DEFAULT, minimum=_WEB_TOP100_TOP_COUNT)
    exhaustive_limit = _env_int(
        "LOTOFACIL_WEB_TOP100_EXHAUSTIVE_LIMIT",
        _WEB_TOP100_EXHAUSTIVE_LIMIT_DEFAULT,
        minimum=_WEB_TOP100_TOP_COUNT,
    )
    max_overlap = _env_int("LOTOFACIL_WEB_TOP100_MAX_OVERLAP", _WEB_TOP100_MAX_OVERLAP_DEFAULT, minimum=8)
    return {
        "top_count": int(_WEB_TOP100_TOP_COUNT),
        "top_pool": int(top_pool),
        "max_overlap": int(min(15, max_overlap)),
        "exhaustive_limit": int(exhaustive_limit),
    }


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
          {key: 'estrategia_origem_top100', label: 'Estratégia'},
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
          {key: 'estrategia_origem_top100', label: 'Estratégia'},
          {key: 'primeira_dezena_top100', label: 'Primeira dezena'},
          {key: 'score_top100', label: 'Score Top 100'},
          {key: 'score_top50_refinado', label: 'Score refinado'},
          {key: 'refinador_top50_aplicado', label: 'Refinador'},
          {key: 'criterio_top100', label: 'Critério'}
        ]) + '</section>' +
      '</section>';
    }
    function renderTop100Job(data) {
      return '<section class="calibration-panel">' +
        '<div class="game-head"><h2>Geração Top 100</h2><span class="tag">' + escapeHtml(data.status || 'sem status') + '</span></div>' +
        '<div class="status-grid">' +
          statusCard('Status', data.status || '-') +
          statusCard('Iniciado em', data.started_at || '-') +
          statusCard('Finalizado em', data.finished_at || '-') +
          statusCard('Concurso alvo', data.concurso_alvo || '-') +
          statusCard('Data concurso', data.data_proximo_concurso || '-') +
          statusCard('Jogos gerados', data.selected_rows || '-') +
        '</div>' +
        '<div class="exclusives"><strong>Como ler:</strong> a geração roda em segundo plano. Você pode deixar esta tela aberta; quando terminar, os 100 jogos aparecem automaticamente.</div>' +
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
    function scoreDictRows(scores) {
      if (!scores || typeof scores !== 'object') {
        return [];
      }
      return Object.keys(scores).map(function(key) {
        return {dezena: key, score: Number(scores[key] || 0)};
      }).sort(function(a, b) { return b.score - a.score; }).slice(0, 12);
    }
    function renderTop100Repair(data) {
      const state = data.state || {};
      return '<section class="calibration-panel">' +
        '<div class="game-head"><h2>Reparo Top100 por quase-acerto</h2><span class="tag">' + escapeHtml(state.status || 'sem status') + '</span></div>' +
        '<div class="status-grid">' +
          statusCard('Concursos aprendidos', state.learned_contests_count || 0) +
          statusCard('Processados na última execução', state.processed_this_run || 0) +
          statusCard('Melhor quase-acerto visto', state.best_hits_seen || '-') +
          statusCard('Mínimo para aprender', state.min_hits || '-') +
          statusCard('Pendentes', state.pending_prediction_files ?? '-') +
          statusCard('Atualizado em', state.updated_at || '-') +
        '</div>' +
        '<div class="exclusives"><strong>Como ler:</strong> dezenas para entrar são as que faltavam nos quase-acertos históricos; dezenas para sair são falsas positivas frequentes nesses mesmos jogos.</div>' +
        '<section class="comparison"><h2>Dezenas que o reparo tenta colocar</h2>' + tableRows(scoreDictRows(data.add_scores), [
          {key: 'dezena', label: 'Dezena'},
          {key: 'score', label: 'Score'}
        ]) + '</section>' +
        '<section class="comparison"><h2>Dezenas que o reparo tenta remover</h2>' + tableRows(scoreDictRows(data.remove_scores), [
          {key: 'dezena', label: 'Dezena'},
          {key: 'score', label: 'Score'}
        ]) + '</section>' +
      '</section>';
    }
    function renderTop100Walkforward(data) {
      const state = data.state || {};
      const recent = data.recent_results || [];
      const best = data.best_results || [];
      const blocks = data.progress_blocks || [];
      const weights = data.weights || [];
      return '<section class="calibration-panel">' +
        '<div class="game-head"><h2>Aprendizado walk-forward Top100</h2><span class="tag">' + escapeHtml(state.status || 'sem status') + '</span></div>' +
        '<div class="status-grid">' +
          statusCard('Concurso atual', state.current_concurso || '-') +
          statusCard('Último processado', state.last_concurso || '-') +
          statusCard('Concursos aprendidos', state.total_contests_processed || 0) +
          statusCard('Progresso elegível', state.progress_percent !== undefined ? Number(state.progress_percent).toFixed(2) + '%' : '-') +
          statusCard('Pendentes elegíveis', state.remaining_eligible_count ?? '-') +
          statusCard('Próximo pendente', state.next_pending_concurso || '-') +
          statusCard('Melhor acerto médio', state.best_hits_avg ?? '-') +
          statusCard('Melhor acerto máximo', state.best_hits_max ?? '-') +
          statusCard('Hit Top100 exato', state.hit_top100 || 0) +
          statusCard('Hit Top100 %', state.hit_top100_pct !== undefined ? Number(state.hit_top100_pct).toFixed(4) + '%' : '-') +
          statusCard('Rank diag. antes', state.rank_before_avg ?? '-') +
          statusCard('Rank diag. aprendido', state.rank_after_avg ?? '-') +
          statusCard('Último melhor acerto', state.last_best_hits_top100 ?? '-') +
          statusCard('Tempo execução atual', state.elapsed_seconds_current_run ? Number(state.elapsed_seconds_current_run).toFixed(0) + 's' : '-') +
        '</div>' +
        '<div class="exclusives"><strong>Como ler:</strong> para cada concurso encerrado, o motor gera 100 jogos sem saber o resultado, mede o melhor acerto e só depois usa o gabarito para aprender pesos para os próximos concursos.</div>' +
        '<section class="comparison"><h2>Pesos aprendidos para o Top100</h2>' + tableRows(weights, [
          {key: 'feature', label: 'Sinal'},
          {key: 'peso_positivo_percentual', label: 'Peso positivo %'},
          {key: 'peso_penalizador_percentual', label: 'Peso penalizador %'}
        ]) + '</section>' +
        '<section class="comparison"><h2>Evolução Top100 por blocos</h2>' + tableRows(blocks, [
          {key: 'bloco', label: 'Bloco'},
          {key: 'concursos', label: 'Concursos'},
          {key: 'melhor_acerto_medio', label: 'Melhor acerto médio'},
          {key: 'melhor_acerto_maximo', label: 'Máximo'},
          {key: 'rank_antes_medio', label: 'Rank antes'},
          {key: 'rank_aprendido_medio', label: 'Rank aprendido'},
          {key: 'hit_top100_pct', label: 'Hit Top100 %'}
        ]) + '</section>' +
        '<section class="comparison"><h2>Melhores concursos Top100</h2>' + tableRows(best, [
          {key: 'concurso', label: 'Concurso'},
          {key: 'best_hits_top100', label: 'Melhor acerto'},
          {key: 'best_rank_top100', label: 'Rank'},
          {key: 'best_game_top100', label: 'Melhor jogo'},
          {key: 'jogo_real', label: 'Jogo real'}
        ]) + '</section>' +
        '<section class="comparison"><h2>Últimos concursos Top100 aprendidos</h2>' + tableRows(recent, [
          {key: 'concurso', label: 'Concurso'},
          {key: 'best_hits_top100', label: 'Melhor acerto'},
          {key: 'hit_top100', label: 'Acertou 15 exato'},
          {key: 'near_miss_11plus', label: '11+ acertos'},
          {key: 'rank_diagnostico_antes', label: 'Rank antes'},
          {key: 'rank_diagnostico_aprendido', label: 'Rank aprendido'},
          {key: 'jogo_real', label: 'Jogo real'}
        ]) + '</section>' +
      '</section>';
    }
    function renderLearning(data) {
      const unified = data.unified || {};
      const supervised = data.supervised || {};
      const top50 = data.top50 || {};
      const top100Walkforward = data.top100_walkforward || {};
      const top100Repair = data.top100_repair || {};
      const supervisedState = supervised.state || {};
      const top50State = top50.state || {};
      const top100State = top100Walkforward.state || {};
      const repairState = top100Repair.state || {};
      return '<section class="calibration-panel">' +
        '<div class="game-head"><h2>Aprendizado unificado</h2><span class="tag">motor 3.0</span></div>' +
        '<div class="status-grid">' +
          statusCard('Ciclo unificado', unified.status || '-') +
          statusCard('Lock ativo', data.lock_active ? 'sim' : 'não') +
          statusCard('Supervisionado', supervisedState.status || '-') +
          statusCard('Supervisionado progresso', supervisedState.progress_percent !== undefined ? Number(supervisedState.progress_percent).toFixed(2) + '%' : '-') +
          statusCard('Supervisionado rank depois', supervisedState.rank_after_avg ?? '-') +
          statusCard('Top50', top50State.status || '-') +
          statusCard('Top50 progresso', top50State.progress_percent !== undefined ? Number(top50State.progress_percent).toFixed(2) + '%' : '-') +
          statusCard('Top50 rank refinado', top50State.rank_after_avg ?? '-') +
          statusCard('Top50 Hit@50', top50State.hit_top50_after !== undefined ? Number(top50State.hit_top50_after).toFixed(2) + '%' : '-') +
          statusCard('Top50 Hit@100', top50State.hit_top100_after !== undefined ? Number(top50State.hit_top100_after).toFixed(2) + '%' : '-') +
          statusCard('Top100 walk-forward', top100State.status || '-') +
          statusCard('Top100 progresso', top100State.progress_percent !== undefined ? Number(top100State.progress_percent).toFixed(2) + '%' : '-') +
          statusCard('Top100 melhor médio', top100State.best_hits_avg ?? '-') +
          statusCard('Reparo Top100', repairState.status || '-') +
          statusCard('Reparo concursos', repairState.learned_contests_count || 0) +
          statusCard('Reparo melhor quase-acerto', repairState.best_hits_seen || '-') +
        '</div>' +
        '<div class="exclusives"><strong>Como ler:</strong> este botão inicia e acompanha o ciclo completo de aprendizado. Ele roda em blocos pequenos para continuar de onde parou e não travar a interface.</div>' +
      '</section>' +
      renderSupervised(supervised) +
      renderTop50Refinement(top50) +
      renderTop100Walkforward(top100Walkforward) +
      renderTop100Repair(top100Repair);
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
      const data = await request('/api/learning', {method: 'POST'});
      document.getElementById('games').innerHTML = renderLearning(data);
      pollLearning();
    }
    async function pollLearning() {
      const data = await request('/api/learning/status');
      document.getElementById('games').innerHTML = renderLearning(data);
      const unifiedStatus = String((data.unified || {}).status || '');
      if (unifiedStatus === 'running' || unifiedStatus === 'starting' || unifiedStatus === 'already_running' || data.lock_active) {
        window.setTimeout(pollLearning, 10000);
      }
    }
    async function top100() {
      const data = await request('/api/top100', {method: 'POST'});
      if (data.status === 'complete' && Array.isArray(data.games)) {
        document.getElementById('games').innerHTML = renderTop100(data);
        return;
      }
      document.getElementById('games').innerHTML = renderTop100Job(data);
      pollTop100();
    }
    async function pollTop100() {
      const data = await request('/api/top100/status');
      if (data.status === 'complete' && Array.isArray(data.games)) {
        document.getElementById('games').innerHTML = renderTop100(data);
        return;
      }
      document.getElementById('games').innerHTML = renderTop100Job(data);
      if (data.status === 'running' || data.status === 'starting') {
        window.setTimeout(pollTop100, 10000);
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


def _top100_payload_from_file(config: AppConfig) -> dict[str, Any]:
    games = _read_csv_records(config.top100_prediction_csv_path)
    payload: dict[str, Any] = {
        "status": "complete" if games else "idle",
        "games": games,
    }
    if games:
        first = games[0]
        payload.update(
            {
                "concurso_alvo": first.get("concurso_alvo"),
                "data_proximo_concurso": first.get("data_proximo_concurso"),
                "generated_at": first.get("generated_at"),
                "selected_rows": len(games),
                "top_count": len(games),
                "top_pool": first.get("top_pool"),
                "metodo": first.get("metodo") or first.get("source_model") or "top100",
                "prediction_csv_path": str(config.top100_prediction_csv_path),
                "report_path": str(config.top100_prediction_report_path),
                "excel_path": str(config.top100_prediction_excel_path),
            }
        )
    return payload


def _top100_job_snapshot(config: AppConfig) -> dict[str, Any]:
    with _TOP100_JOB_LOCK:
        snapshot = dict(_TOP100_JOB)
    if snapshot.get("summary"):
        summary = snapshot["summary"]
        if isinstance(summary, dict):
            snapshot.update(summary)
    if snapshot.get("status") in {"idle", "complete"}:
        file_payload = _top100_payload_from_file(config)
        if file_payload.get("games"):
            snapshot.update(file_payload)
    return snapshot


def _run_top100_job(config: AppConfig, logger: logging.Logger) -> None:
    settings = _top100_web_settings()
    with _TOP100_JOB_LOCK:
        _TOP100_JOB.update(
            {
                "status": "running",
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "finished_at": None,
                "error": None,
                "summary": None,
                "settings": settings,
            }
        )
    try:
        summary = Top100Pipeline(config=config, logger=logger).predict(
            top_count=settings["top_count"],
            top_pool=settings["top_pool"],
            max_overlap=settings["max_overlap"],
            draw_hour=20,
            draw_minute=0,
            exhaustive_limit=settings["exhaustive_limit"],
            quick_mode=True,
        )
        payload = summary.__dict__.copy()
        payload["settings"] = settings
        payload["games"] = _read_csv_records(config.top100_prediction_csv_path)
        with _TOP100_JOB_LOCK:
            _TOP100_JOB.update(
                {
                    "status": "complete",
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "error": None,
                    "summary": payload,
                }
            )
    except Exception as exc:
        logger.exception("Erro ao gerar Top100 em segundo plano: %s", exc)
        with _TOP100_JOB_LOCK:
            _TOP100_JOB.update(
                {
                    "status": "error",
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "error": str(exc),
                    "summary": None,
                }
            )


def _start_top100_job(config: AppConfig, logger: logging.Logger) -> dict[str, Any]:
    settings = _top100_web_settings()
    with _TOP100_JOB_LOCK:
        if _TOP100_JOB.get("status") == "running":
            return dict(_TOP100_JOB)
        _TOP100_JOB.update(
            {
                "status": "starting",
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "finished_at": None,
                "error": None,
                "summary": None,
                "settings": settings,
            }
        )
    worker = Thread(target=_run_top100_job, args=(config, logger), daemon=True)
    worker.start()
    return _top100_job_snapshot(config)


def _learning_status_payload(config: AppConfig, logger: logging.Logger) -> dict[str, Any]:
    payload = UnifiedLearningPipeline(config=config, logger=logger).status()
    with _LEARNING_JOB_LOCK:
        snapshot = dict(_LEARNING_JOB)
    payload["web_job"] = snapshot
    if snapshot.get("status") in {"running", "starting"}:
        unified = payload.get("unified", {}) if isinstance(payload.get("unified"), dict) else {}
        unified["status"] = snapshot.get("status")
        unified["web_started_at"] = snapshot.get("started_at")
        payload["unified"] = unified
    return payload


def _run_learning_job(config: AppConfig, logger: logging.Logger) -> None:
    with _LEARNING_JOB_LOCK:
        _LEARNING_JOB.update(
            {
                "status": "running",
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "finished_at": None,
                "error": None,
                "summary": None,
            }
        )
    try:
        summary = UnifiedLearningPipeline(config=config, logger=logger).run_once(
            seed=123,
            draw_hour=20,
            draw_minute=0,
            reset=False,
        )
        with _LEARNING_JOB_LOCK:
            _LEARNING_JOB.update(
                {
                    "status": "complete",
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "error": None,
                    "summary": summary.__dict__.copy(),
                }
            )
    except Exception as exc:
        logger.exception("Erro ao rodar aprendizado unificado em segundo plano: %s", exc)
        with _LEARNING_JOB_LOCK:
            _LEARNING_JOB.update(
                {
                    "status": "error",
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "error": str(exc),
                    "summary": None,
                }
            )


def _start_learning_job(config: AppConfig, logger: logging.Logger) -> dict[str, Any]:
    already_running = False
    with _LEARNING_JOB_LOCK:
        if _LEARNING_JOB.get("status") in {"running", "starting"}:
            already_running = True
        else:
            _LEARNING_JOB.update(
                {
                    "status": "starting",
                    "started_at": datetime.now().isoformat(timespec="seconds"),
                    "finished_at": None,
                    "error": None,
                    "summary": None,
                }
            )
    if already_running:
        return _learning_status_payload(config, logger)
    worker = Thread(target=_run_learning_job, args=(config, logger), daemon=True)
    worker.start()
    return _learning_status_payload(config, logger)


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
                self._handle_json(lambda: _learning_status_payload(config, logger))
                return
            if self.path == "/api/top100/status":
                self._handle_json(lambda: _top100_job_snapshot(config))
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
                self._handle_json(lambda: _start_top100_job(config, logger))
                return
            if self.path == "/api/learning":
                self._handle_json(lambda: _start_learning_job(config, logger))
                return
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "rota nao encontrada"})

        def _handle_json(self, fn: Callable[[], dict]) -> None:
            try:
                payload = fn()
                serializable = _json_safe(payload)
                _json_response(self, HTTPStatus.OK, serializable)
            except BrokenPipeError:
                logger.warning("Cliente fechou a conexao antes da resposta JSON.")
            except Exception as exc:
                logger.exception("Erro na interface web: %s", exc)
                try:
                    _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
                except BrokenPipeError:
                    logger.warning("Cliente fechou a conexao antes da resposta de erro.")

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
