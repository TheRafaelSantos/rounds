from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .config import AppConfig
from .context_features import build_target_context
from .normalize import DEZENAS
from .storage import list_raw_files, load_raw_payload, sanitize_dataframe_for_tabular_output


@dataclass(frozen=True)
class ExportSummary:
    excel_path: str
    sheets: List[str]

    def to_console(self) -> str:
        return "\n".join(
            [
                "",
                "Resumo Lotofacil Analytics - Export",
                f"Excel consolidado: {self.excel_path}",
                f"Abas: {', '.join(self.sheets)}",
                "Mensagem: Exportacao consolidada gerada com as abas do briefing.",
            ]
        )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _read_csv_many(directory: Path, pattern: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(directory.glob(pattern)):
        df = _read_csv(path)
        if df.empty:
            continue
        df = df.copy()
        df.insert(0, "arquivo_origem", path.name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _empty_sheet(message: str) -> pd.DataFrame:
    return pd.DataFrame([{"observacao": message}])


def _raw_payloads(config: AppConfig) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in list_raw_files(config.raw_dir):
        payload = load_raw_payload(path)
        record = pd.json_normalize(payload, sep=".").iloc[0].to_dict()
        record["raw_json_file"] = str(path.relative_to(config.base_dir))
        for key, value in list(record.items()):
            if isinstance(value, (list, dict)):
                record[key] = json.dumps(value, ensure_ascii=False)
        rows.append(record)
    return pd.DataFrame(rows) if rows else _empty_sheet("Nenhum JSON bruto encontrado. Rode python main.py --update.")


def _concursos_view(concursos: pd.DataFrame) -> pd.DataFrame:
    if concursos.empty:
        return _empty_sheet("Historico tratado nao encontrado.")
    out = pd.DataFrame()
    out["concurso"] = concursos["concurso"]
    out["data_sorteio"] = concursos["data_sorteio"]
    out["dia_semana"] = pd.to_datetime(concursos["data_sorteio"], errors="coerce").dt.dayofweek + 1
    out["tipo_concurso"] = concursos.get("tipo_jogo")
    out["status"] = "normalizado"
    out["acumulado"] = concursos.get("acumulado")
    out["local_sorteio"] = concursos.get("local_sorteio")
    out["cidade_sorteio"] = concursos.get("cidade_sorteio")
    out["uf_sorteio"] = concursos.get("uf_sorteio")
    for idx, col in enumerate(DEZENAS, start=1):
        out[f"n{idx}"] = concursos[col]
    for acertos in [15, 14, 13, 12, 11]:
        out[f"ganhadores_{acertos}_acertos"] = concursos.get(f"ganhadores_{acertos}")
        out[f"premio_{acertos}_acertos"] = concursos.get(f"premio_{acertos}")
    out["valor_arrecadado"] = concursos.get("valor_arrecadado")
    out["valor_acumulado_proximo_concurso"] = concursos.get("valor_acumulado_proximo_concurso")
    out["valor_estimado_proximo_concurso"] = concursos.get("valor_estimado_proximo_concurso")
    out["data_proximo_concurso"] = concursos.get("data_proximo_concurso")
    out["indicador_concurso_especial"] = concursos.get("indicador_concurso_especial")
    return out


def _frequencias(dezenas_historico: pd.DataFrame) -> pd.DataFrame:
    if dezenas_historico.empty:
        return _empty_sheet("Historico por dezena nao encontrado. Rode python main.py --dezenas.")
    latest = dezenas_historico.sort_values("concurso").groupby("dezena", as_index=False).tail(1)
    cols = [
        "dezena",
        "freq_total_ate_anterior",
        "freq_dezena_ultimos_5",
        "freq_dezena_ultimos_10",
        "freq_dezena_ultimos_20",
        "freq_dezena_ultimos_50",
        "freq_dezena_ultimos_100",
        "freq_dezena_ultimos_250",
        "percentual_freq_total",
    ]
    return latest[[col for col in cols if col in latest.columns]].sort_values("dezena")


def _atrasos(dezenas_historico: pd.DataFrame) -> pd.DataFrame:
    if dezenas_historico.empty:
        return _empty_sheet("Historico por dezena nao encontrado. Rode python main.py --dezenas.")
    latest = dezenas_historico.sort_values("concurso").groupby("dezena", as_index=False).tail(1)
    cols = [
        "dezena",
        "atraso_atual_dezena",
        "maior_atraso_historico_dezena",
        "media_atraso_dezena",
        "mediana_atraso_dezena",
        "desvio_atraso_dezena",
        "percentil_atraso_dezena",
        "ranking_atraso_dezena",
    ]
    return latest[[col for col in cols if col in latest.columns]].sort_values("dezena")


def _rankings(dezenas_historico: pd.DataFrame) -> pd.DataFrame:
    if dezenas_historico.empty:
        return _empty_sheet("Rankings nao encontrados. Rode python main.py --dezenas.")
    cols = [
        "concurso",
        "data_sorteio",
        "dezena",
        "rank_freq_total",
        "rank_freq_ultimos_5",
        "rank_freq_ultimos_10",
        "rank_freq_ultimos_20",
        "rank_freq_ultimos_50",
        "rank_freq_ultimos_100",
        "rank_atraso",
        "rank_score_quente",
        "rank_score_frio",
        "rank_score_equilibrado",
    ]
    return dezenas_historico[[col for col in cols if col in dezenas_historico.columns]]


def _parametros(config: AppConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"parametro": "gerado_em", "valor": datetime.now().isoformat(timespec="seconds")},
            {"parametro": "base_dir", "valor": str(config.base_dir)},
            {"parametro": "fonte_dados", "valor": "https://servicebus2.caixa.gov.br/portaldeloterias/api/lotofacil"},
            {"parametro": "comando_full", "valor": "python main.py --full"},
            {"parametro": "comando_update", "valor": "python main.py --update"},
            {"parametro": "comando_features", "valor": "python main.py --features"},
            {"parametro": "comando_transitions", "valor": "python main.py --transitions"},
            {"parametro": "comando_backtest", "valor": "python main.py --backtest"},
            {"parametro": "comando_export", "valor": "python main.py --export"},
            {"parametro": "comando_optimize_exaustivo", "valor": "python main.py --optimize --engine exaustivo --top-games 5000 --draw-hour 20 --draw-minute 0"},
            {"parametro": "comando_generate_games", "valor": "python main.py --generate-games --method balanceado_basico --qty 10"},
            {"parametro": "comando_predict", "valor": "python main.py --predict"},
            {"parametro": "comando_predict_exaustivo", "valor": "python main.py --predict --engine exaustivo --draw-hour 20 --draw-minute 0"},
            {"parametro": "comando_predict_completo", "valor": "python main.py --predict --mode completo"},
            {"parametro": "comando_predict_single", "valor": "python main.py --predict-single --engine exaustivo --draw-hour 20 --draw-minute 0"},
            {"parametro": "comando_backtest_exhaustive", "valor": "python main.py --backtest-exhaustive --validation-n-eval 3 --min-history 300"},
            {"parametro": "comando_ablation_test", "valor": "python main.py --ablation-test --validation-n-eval 3 --min-history 300"},
            {"parametro": "comando_tune_weights", "valor": "python main.py --tune-weights --validation-n-eval 3 --min-history 300"},
            {"parametro": "comando_analyze_result", "valor": "python main.py --analyze-result --result-label exemplo --actual-numbers \"01 02 03 04 05 06 07 08 09 10 11 12 13 14 15\""},
            {"parametro": "comando_final_backtest", "valor": "python main.py --final-backtest"},
            {"parametro": "comando_serve", "valor": "python main.py --serve"},
            {"parametro": "comando_build_exe", "valor": "python main.py --build-exe"},
        ]
    )


def _logs_execucao(config: AppConfig) -> pd.DataFrame:
    path = config.logs_dir / "lotofacil_analytics.log"
    if not path.exists():
        return _empty_sheet("Log local nao encontrado.")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-5000:]
    return pd.DataFrame([{"linha": idx + 1, "log": line} for idx, line in enumerate(lines)])


def _contexto_proximo_concurso(concursos: pd.DataFrame) -> pd.DataFrame:
    if concursos.empty:
        return _empty_sheet("Historico tratado nao encontrado.")
    context = build_target_context(concursos)
    return pd.DataFrame([context.as_dict()])


def export_full_workbook(config: AppConfig, logger: logging.Logger) -> ExportSummary:
    concursos = _read_csv(config.processed_csv_path)
    features = _read_csv(config.features_base_csv_path)
    dezenas_long = _read_csv(config.dezenas_long_csv_path)
    dezenas_historico = _read_csv(config.dezenas_historico_csv_path)
    pares = _read_csv(config.combinacoes_pares_csv_path)
    trios = _read_csv(config.combinacoes_trios_csv_path)
    quartetos = _read_csv(config.combinacoes_quartetos_csv_path)
    transitions = _read_csv(config.transition_csv_path)
    transitions_summary = _read_csv(config.transition_summary_csv_path)
    transitions_dezenas = _read_csv(config.transition_dezenas_csv_path)
    backtest = _read_csv(config.backtest_csv_path)
    generated_games = _read_csv(config.generated_games_csv_path)
    prediction = _read_csv(config.prediction_csv_path)
    single_prediction = _read_csv(config.single_prediction_csv_path)
    post_result_games = _read_csv_many(config.processed_dir, "lotofacil_pos_sorteio_jogos*.csv")
    post_result_dezenas = _read_csv_many(config.processed_dir, "lotofacil_pos_sorteio_dezenas*.csv")
    final_backtest = _read_csv(config.final_backtest_csv_path)
    final_backtest_summary = _read_csv(config.final_backtest_summary_csv_path)
    exhaustive_backtest = _read_csv(config.exhaustive_backtest_csv_path)
    exhaustive_backtest_summary = _read_csv(config.exhaustive_backtest_summary_csv_path)
    ablation = _read_csv(config.ablation_csv_path)
    ablation_summary = _read_csv(config.ablation_summary_csv_path)
    tune_weights = _read_csv(config.tune_weights_csv_path)
    tune_weights_summary = _read_csv(config.tune_weights_summary_csv_path)
    jogos_gerados = generated_games if not generated_games.empty else prediction

    sheets: Dict[str, pd.DataFrame] = {
        "concursos_raw": _raw_payloads(config),
        "concursos": _concursos_view(concursos),
        "concursos_features": features if not features.empty else _empty_sheet("Features nao encontradas. Rode python main.py --features."),
        "dezenas_long": dezenas_long if not dezenas_long.empty else _empty_sheet("dezenas_long nao encontrado. Rode python main.py --dezenas."),
        "dezenas_historico": dezenas_historico if not dezenas_historico.empty else _empty_sheet("dezenas_historico nao encontrado. Rode python main.py --dezenas."),
        "frequencias": _frequencias(dezenas_historico),
        "atrasos": _atrasos(dezenas_historico),
        "pares": pares if not pares.empty else _empty_sheet("Pares nao encontrados. Rode python main.py --combinacoes."),
        "trios": trios if not trios.empty else _empty_sheet("Trios nao encontrados. Rode python main.py --combinacoes."),
        "quartetos": quartetos if not quartetos.empty else _empty_sheet("Quartetos nao encontrados. Rode python main.py --combinacoes."),
        "rankings": _rankings(dezenas_historico),
        "transicoes": transitions if not transitions.empty else _empty_sheet("Transicoes nao encontradas. Rode python main.py --transitions."),
        "transicoes_resumo": transitions_summary if not transitions_summary.empty else _empty_sheet("Resumo de transicoes nao encontrado. Rode python main.py --transitions."),
        "transicoes_dezenas": transitions_dezenas if not transitions_dezenas.empty else _empty_sheet("Transicoes por dezena nao encontradas. Rode python main.py --transitions."),
        "backtest": backtest if not backtest.empty else _empty_sheet("Backtest nao encontrado. Rode python main.py --backtest."),
        "backtest_score_final": final_backtest if not final_backtest.empty else _empty_sheet("Backtest score final nao encontrado. Rode python main.py --final-backtest."),
        "backtest_score_final_resumo": final_backtest_summary if not final_backtest_summary.empty else _empty_sheet("Resumo do backtest score final nao encontrado. Rode python main.py --final-backtest."),
        "jogo_unico": single_prediction if not single_prediction.empty else _empty_sheet("Jogo unico nao encontrado. Rode python main.py --predict-single."),
        "backtest_exaustivo": exhaustive_backtest if not exhaustive_backtest.empty else _empty_sheet("Backtest exaustivo nao encontrado. Rode python main.py --backtest-exhaustive."),
        "backtest_exaustivo_resumo": exhaustive_backtest_summary if not exhaustive_backtest_summary.empty else _empty_sheet("Resumo do backtest exaustivo nao encontrado. Rode python main.py --backtest-exhaustive."),
        "ablation_test": ablation if not ablation.empty else _empty_sheet("Ablation test nao encontrado. Rode python main.py --ablation-test."),
        "ablation_test_resumo": ablation_summary if not ablation_summary.empty else _empty_sheet("Resumo do ablation test nao encontrado. Rode python main.py --ablation-test."),
        "tune_weights": tune_weights if not tune_weights.empty else _empty_sheet("Tune weights nao encontrado. Rode python main.py --tune-weights."),
        "tune_weights_resumo": tune_weights_summary if not tune_weights_summary.empty else _empty_sheet("Resumo do tune weights nao encontrado. Rode python main.py --tune-weights."),
        "jogos_gerados": jogos_gerados if not jogos_gerados.empty else _empty_sheet("Jogos nao encontrados. Rode python main.py --generate-games ou --predict."),
        "pos_sorteio_jogos": post_result_games if not post_result_games.empty else _empty_sheet("Analise pos-sorteio nao encontrada. Rode python main.py --analyze-result."),
        "pos_sorteio_dezenas": post_result_dezenas if not post_result_dezenas.empty else _empty_sheet("Auditoria de dezenas pos-sorteio nao encontrada. Rode python main.py --analyze-result."),
        "contexto_proximo_concurso": _contexto_proximo_concurso(concursos),
        "parametros": _parametros(config),
        "logs_execucao": _logs_execucao(config),
    }

    config.full_export_excel_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = config.full_export_excel_path.with_name(f"{config.full_export_excel_path.stem}.tmp{config.full_export_excel_path.suffix}")
    if temp_path.exists():
        temp_path.unlink()
    with pd.ExcelWriter(temp_path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            safe_df = sanitize_dataframe_for_tabular_output(df)
            safe_df.to_excel(writer, index=False, sheet_name=sheet_name)
    temp_path.replace(config.full_export_excel_path)

    logger.info("Excel consolidado salvo em %s", config.full_export_excel_path)
    return ExportSummary(excel_path=str(config.full_export_excel_path), sheets=list(sheets.keys()))
