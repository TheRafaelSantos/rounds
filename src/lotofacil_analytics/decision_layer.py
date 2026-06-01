from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import pandas as pd

from .backtest_lotofacil import nums_from_row
from .context_features import TargetContext, build_target_context
from .exhaustive_optimizer import (
    DEFAULT_EXHAUSTIVE_WEIGHTS,
    EXHAUSTIVE_SOURCE_MODEL,
    TOTAL_COMBINATIONS,
    build_exhaustive_candidates,
    format_exhaustive_weights,
    resolve_exhaustive_weights,
)
from .optimizer import build_optimized_candidates
from .post_result_analysis import format_nums, parse_numbers
from .predictor import AVISO_TECNICO


SOURCE_MODEL_EXAUSTIVO = EXHAUSTIVE_SOURCE_MODEL
SOURCE_MODEL_SINGLE = "ensemble_score_v4_exaustivo_transicao_decision_layer_single"

WEIGHT_PROFILE_PRESETS: Dict[str, Dict[str, float]] = {
    "padrao_atual": dict(DEFAULT_EXHAUSTIVE_WEIGHTS),
    "contexto_forte": {
        "estatistico": 0.14,
        "historico": 0.09,
        "atraso": 0.05,
        "combinatorio": 0.09,
        "localidade_numerologia": 0.32,
        "cenarios": 0.10,
        "contrarian": 0.07,
        "transicao": 0.09,
        "nao_repeticao_exata": 0.05,
    },
    "historico_forte": {
        "estatistico": 0.15,
        "historico": 0.23,
        "atraso": 0.08,
        "combinatorio": 0.11,
        "localidade_numerologia": 0.14,
        "cenarios": 0.09,
        "contrarian": 0.05,
        "transicao": 0.10,
        "nao_repeticao_exata": 0.05,
    },
    "combinatorio_forte": {
        "estatistico": 0.15,
        "historico": 0.10,
        "atraso": 0.05,
        "combinatorio": 0.26,
        "localidade_numerologia": 0.14,
        "cenarios": 0.08,
        "contrarian": 0.07,
        "transicao": 0.10,
        "nao_repeticao_exata": 0.05,
    },
    "contrarian_forte": {
        "estatistico": 0.14,
        "historico": 0.09,
        "atraso": 0.05,
        "combinatorio": 0.09,
        "localidade_numerologia": 0.14,
        "cenarios": 0.10,
        "contrarian": 0.24,
        "transicao": 0.10,
        "nao_repeticao_exata": 0.05,
    },
    "estatistico_forte": {
        "estatistico": 0.30,
        "historico": 0.09,
        "atraso": 0.05,
        "combinatorio": 0.11,
        "localidade_numerologia": 0.13,
        "cenarios": 0.10,
        "contrarian": 0.08,
        "transicao": 0.09,
        "nao_repeticao_exata": 0.05,
    },
    "cenarios_forte": {
        "estatistico": 0.14,
        "historico": 0.09,
        "atraso": 0.05,
        "combinatorio": 0.09,
        "localidade_numerologia": 0.14,
        "cenarios": 0.28,
        "contrarian": 0.06,
        "transicao": 0.10,
        "nao_repeticao_exata": 0.05,
    },
    "atraso_forte": {
        "estatistico": 0.15,
        "historico": 0.12,
        "atraso": 0.20,
        "combinatorio": 0.09,
        "localidade_numerologia": 0.14,
        "cenarios": 0.09,
        "contrarian": 0.06,
        "transicao": 0.10,
        "nao_repeticao_exata": 0.05,
    },
    "transicao_forte": {
        "estatistico": 0.14,
        "historico": 0.10,
        "atraso": 0.05,
        "combinatorio": 0.09,
        "localidade_numerologia": 0.15,
        "cenarios": 0.10,
        "contrarian": 0.07,
        "transicao": 0.25,
        "nao_repeticao_exata": 0.05,
    },
}

ABLATION_VARIANTS: List[Tuple[str, str | None]] = [
    ("completo", None),
    ("sem_estatistico", "estatistico"),
    ("sem_historico", "historico"),
    ("sem_atraso", "atraso"),
    ("sem_combinatorio", "combinatorio"),
    ("sem_lua_local_numerologia", "localidade_numerologia"),
    ("sem_cenarios", "cenarios"),
    ("sem_contrarian", "contrarian"),
    ("sem_transicao", "transicao"),
]


@dataclass(frozen=True)
class SinglePredictionSummary:
    concurso_alvo: int
    generated_at: str
    data_proximo_concurso: str
    dia_semana: str
    fase_lua: str
    iluminacao_lua_percentual: float
    numerologia_data_raiz: int
    weight_profile: str
    jogo_unico: str
    score_final: float
    prediction_csv_path: str
    report_path: str
    excel_path: str

    def to_console(self) -> str:
        return "\n".join(
            [
                "",
                "Resumo Lotofacil Analytics - Camada de Decisao / Jogo Unico",
                f"Concurso-alvo: {self.concurso_alvo}",
                f"Gerado em: {self.generated_at}",
                f"Data do proximo concurso: {self.data_proximo_concurso}",
                f"Dia da semana: {self.dia_semana}",
                f"Lua no horario de Brasilia: {self.fase_lua} ({self.iluminacao_lua_percentual:.2f}% iluminada)",
                f"Numerologia da data: raiz {self.numerologia_data_raiz}",
                f"Perfil de pesos: {self.weight_profile}",
                f"Jogo unico principal: {self.jogo_unico}",
                f"Score final: {self.score_final:.6f}",
                f"Aviso: {AVISO_TECNICO}",
                f"CSV: {self.prediction_csv_path}",
                f"Relatorio tecnico: {self.report_path}",
                f"Excel: {self.excel_path}",
            ]
        )


@dataclass(frozen=True)
class ExhaustiveValidationSummary:
    action: str
    rows: int
    contests: int
    first_concurso: int
    last_concurso: int
    average_hits: float
    best_hits: int
    best_profile: str
    results_csv_path: str
    summary_csv_path: str
    excel_path: str
    extra_path: str | None = None

    def to_console(self) -> str:
        lines = [
            "",
            f"Resumo Lotofacil Analytics - {self.action}",
            f"Concursos avaliados: {self.contests}",
            f"Primeiro concurso avaliado: {self.first_concurso}",
            f"Ultimo concurso avaliado: {self.last_concurso}",
            f"Linhas de resultado: {self.rows}",
            f"Media de acertos do jogo unico: {self.average_hits:.6f}",
            f"Maior acerto: {self.best_hits}",
            f"Melhor perfil: {self.best_profile}",
            f"CSV resultados: {self.results_csv_path}",
            f"CSV resumo: {self.summary_csv_path}",
            f"Excel: {self.excel_path}",
        ]
        if self.extra_path:
            lines.append(f"Arquivo auxiliar: {self.extra_path}")
        lines.append("Mensagem: Camada superior gerada sem remover as analises atuais.")
        return "\n".join(lines)


def weights_for_profile(profile_name: str) -> Dict[str, float]:
    if profile_name not in WEIGHT_PROFILE_PRESETS:
        valid = ", ".join(sorted(WEIGHT_PROFILE_PRESETS))
        raise ValueError(f"Perfil de pesos desconhecido: {profile_name}. Perfis validos: {valid}.")
    return resolve_exhaustive_weights(WEIGHT_PROFILE_PRESETS[profile_name])


def _disable_component(weights: Mapping[str, float], component: str) -> Dict[str, float]:
    resolved = resolve_exhaustive_weights(weights)
    if component not in resolved:
        valid = ", ".join(sorted(resolved))
        raise ValueError(f"Componente de ablation desconhecido: {component}. Componentes validos: {valid}.")
    adjusted = dict(resolved)
    adjusted[component] = 0.0
    return resolve_exhaustive_weights(adjusted)


def _summary_value(summary: pd.DataFrame, metric: str, default: object = None) -> object:
    if summary.empty or "metrica" not in summary.columns or "valor" not in summary.columns:
        return default
    values = summary.loc[summary["metrica"] == metric, "valor"]
    if values.empty:
        return default
    return values.iloc[0]


def _best_candidate(candidates: pd.DataFrame) -> pd.Series:
    if candidates.empty:
        raise ValueError("Nenhum candidato disponivel para selecionar o jogo unico.")
    if "nums" not in candidates.columns or "score_final" not in candidates.columns:
        raise ValueError("Tabela de candidatos precisa ter colunas nums e score_final.")
    ranked = candidates.copy().sort_values(["score_final", "nums"], ascending=[False, True]).reset_index(drop=True)
    best = ranked.iloc[0].copy()
    parse_numbers(str(best["nums"]))
    return best


def _has_default_exhaustive_candidates(existing_candidates: pd.DataFrame | None) -> bool:
    if existing_candidates is None or existing_candidates.empty:
        return False
    required = {"nums", "score_final", "source_model", "score_contextual", "contexto_fase_lua", "score_transicao"}
    if not required.issubset(existing_candidates.columns):
        return False
    if "total_combinacoes_avaliadas" not in existing_candidates.columns:
        return False
    max_evaluated = pd.to_numeric(existing_candidates["total_combinacoes_avaliadas"], errors="coerce").max()
    if pd.isna(max_evaluated) or int(max_evaluated) < TOTAL_COMBINATIONS:
        return False
    return str(existing_candidates["source_model"].iloc[0]) == SOURCE_MODEL_EXAUSTIVO


def _candidate_rows_to_single_output(
    *,
    best: pd.Series,
    generated_at: str,
    concurso_alvo: int,
    target_context: TargetContext,
    weight_profile: str,
    score_weights: Mapping[str, float],
) -> pd.DataFrame:
    row = best.to_dict()
    row["source_model"] = SOURCE_MODEL_SINGLE
    row["metodo"] = SOURCE_MODEL_SINGLE
    row["jogo"] = 1
    row["generated_at"] = generated_at
    row["concurso_alvo"] = int(concurso_alvo)
    row["data_proximo_concurso"] = target_context.data_proximo_concurso
    row["horario_brasilia_assumido"] = target_context.horario_brasilia_assumido
    row["dia_semana_proximo_concurso"] = target_context.dia_semana_nome
    row["fase_lua_proximo_concurso"] = target_context.fase_lua
    row["iluminacao_lua_percentual"] = target_context.iluminacao_lua_percentual
    row["numerologia_data_raiz"] = target_context.numerologia_data_raiz
    row["numerologia_concurso_raiz"] = target_context.numerologia_concurso_raiz
    row["numerologia_dia_mes_raiz"] = target_context.numerologia_dia_mes_raiz
    row["local_sorteio_assumido"] = target_context.local_sorteio_assumido
    row["cidade_sorteio_assumida"] = target_context.cidade_sorteio_assumida
    row["uf_sorteio_assumida"] = target_context.uf_sorteio_assumida
    row["bairro_sorteio_assumido"] = target_context.bairro_sorteio_assumido
    row["weight_profile"] = weight_profile
    row["score_weights"] = format_exhaustive_weights(score_weights)
    row["aviso"] = AVISO_TECNICO
    preferred = [
        "generated_at",
        "concurso_alvo",
        "data_proximo_concurso",
        "horario_brasilia_assumido",
        "dia_semana_proximo_concurso",
        "fase_lua_proximo_concurso",
        "iluminacao_lua_percentual",
        "numerologia_data_raiz",
        "numerologia_concurso_raiz",
        "numerologia_dia_mes_raiz",
        "local_sorteio_assumido",
        "cidade_sorteio_assumida",
        "uf_sorteio_assumida",
        "bairro_sorteio_assumido",
        "weight_profile",
        "score_weights",
        "jogo",
        "nums",
        "score_final",
        "score_estatistico",
        "score_historico",
        "score_atraso",
        "score_combinatorio",
        "score_contextual",
        "score_localidade_numerologia",
        "score_cenarios",
        "score_contrarian",
        "score_transicao",
        "source_model",
        "metodo",
        "aviso",
    ]
    cols = [col for col in preferred if col in row] + [col for col in row if col not in preferred]
    return pd.DataFrame([{col: row[col] for col in cols}])


def build_single_prediction_report(
    *,
    output: pd.DataFrame,
    target_context: TargetContext,
    generated_at: str,
    last_concurso: int,
    last_date: str,
    weight_profile: str,
    score_weights: Mapping[str, float],
) -> str:
    row = output.iloc[0]
    lines = [
        "# Relatorio tecnico - Camada de decisao / jogo unico",
        "",
        f"Gerado em: {generated_at}",
        f"Ultimo concurso na base: {last_concurso} ({last_date})",
        f"Concurso-alvo estimado: {target_context.concurso_alvo}",
        f"Metodo: {SOURCE_MODEL_SINGLE}",
        f"Perfil de pesos: {weight_profile}",
        f"Pesos normalizados: {format_exhaustive_weights(score_weights)}",
        "",
        "## Contexto do proximo concurso",
        "",
        f"- Data do proximo concurso: {target_context.data_proximo_concurso}",
        f"- Horario de Brasilia usado no calculo lunar: {target_context.horario_brasilia_assumido}",
        f"- Dia da semana: {target_context.dia_semana_nome}",
        f"- Periodo do ano: {target_context.estacao_do_ano}; trimestre {target_context.trimestre}; semestre {target_context.semestre}",
        f"- Lua: {target_context.fase_lua}; idade {target_context.idade_lua:.2f} dias; iluminacao {target_context.iluminacao_lua_percentual:.2f}%",
        f"- Numerologia: raiz da data {target_context.numerologia_data_raiz}; raiz do concurso {target_context.numerologia_concurso_raiz}; raiz dia+mes {target_context.numerologia_dia_mes_raiz}",
        f"- Localidade usada: {target_context.local_sorteio_assumido or '-'} | {target_context.cidade_sorteio_assumida or '-'} | {target_context.uf_sorteio_assumida or '-'}",
        f"- Bairro: {target_context.bairro_sorteio_assumido or 'indisponivel na base atual'}",
        f"- Observacao localidade: {target_context.observacao_localidade}",
        "",
        "## Jogo unico selecionado",
        "",
        f"- Jogo unico: {row['nums']} | score_final={float(row['score_final']):.6f} | score_transicao={float(row.get('score_transicao', 0)):.6f}",
        "",
        "## O que esta camada faz",
        "",
        "1. Mantem o score exaustivo atual como base.",
        "2. Seleciona apenas o melhor jogo completo de 15 dezenas.",
        "3. Explicita o perfil de pesos usado.",
        "4. Mantem lua, numerologia, localidade, periodo do ano, dia da semana, historico, atrasos, combinacoes, cenarios e contrarian no score.",
        "5. Adiciona transicao historica concurso a concurso: repetidas, entradas, saidas e mudanca estrutural.",
        "6. Nao divide a previsao entre dois jogos.",
        "",
        f"Combinacoes possiveis avaliaveis da Lotofacil: {TOTAL_COMBINATIONS}.",
        f"Aviso: {AVISO_TECNICO}",
    ]
    return "\n".join(lines) + "\n"


def build_single_prediction(
    concursos: pd.DataFrame,
    *,
    existing_candidates: pd.DataFrame | None,
    seed: int,
    candidate_pool: int,
    top_games: int,
    generations: int,
    population: int,
    draw_hour: int,
    draw_minute: int,
    engine: str,
    exhaustive_limit: int | None,
    weight_profile: str,
    prediction_csv_path: Path,
    report_path: Path,
    excel_path: Path,
) -> SinglePredictionSummary:
    if concursos.empty:
        raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")

    df = concursos.copy().sort_values("concurso").reset_index(drop=True)
    last = df.iloc[-1]
    last_concurso = int(last["concurso"])
    last_date = str(last["data_sorteio"])
    target_context = build_target_context(df, draw_hour=draw_hour, draw_minute=draw_minute)
    weights = weights_for_profile(weight_profile)

    if engine == "exaustivo":
        if weight_profile == "padrao_atual" and exhaustive_limit is None and _has_default_exhaustive_candidates(existing_candidates):
            candidates = existing_candidates.copy()
        else:
            candidates, _summary = build_exhaustive_candidates(
                df,
                top_games=max(1, int(top_games)),
                draw_hour=draw_hour,
                draw_minute=draw_minute,
                limit_combinations=exhaustive_limit,
                weights=weights,
            )
    else:
        candidates, _summary = build_optimized_candidates(
            df,
            seed=seed,
            candidate_pool=candidate_pool,
            top_games=max(1, int(top_games)),
            generations=generations,
            population=population,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
        )

    best = _best_candidate(candidates)
    generated_at = datetime.now().isoformat(timespec="seconds")
    output = _candidate_rows_to_single_output(
        best=best,
        generated_at=generated_at,
        concurso_alvo=target_context.concurso_alvo,
        target_context=target_context,
        weight_profile=weight_profile,
        score_weights=weights,
    )

    prediction_csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(prediction_csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        output.to_excel(writer, index=False, sheet_name="jogo_unico")
    report_path.write_text(
        build_single_prediction_report(
            output=output,
            target_context=target_context,
            generated_at=generated_at,
            last_concurso=last_concurso,
            last_date=last_date,
            weight_profile=weight_profile,
            score_weights=weights,
        ),
        encoding="utf-8",
    )

    return SinglePredictionSummary(
        concurso_alvo=target_context.concurso_alvo,
        generated_at=generated_at,
        data_proximo_concurso=target_context.data_proximo_concurso,
        dia_semana=target_context.dia_semana_nome,
        fase_lua=target_context.fase_lua,
        iluminacao_lua_percentual=target_context.iluminacao_lua_percentual,
        numerologia_data_raiz=target_context.numerologia_data_raiz,
        weight_profile=weight_profile,
        jogo_unico=str(output.loc[0, "nums"]),
        score_final=float(output.loc[0, "score_final"]),
        prediction_csv_path=str(prediction_csv_path),
        report_path=str(report_path),
        excel_path=str(excel_path),
    )


def _result_row(
    *,
    profile_name: str,
    weights: Mapping[str, float],
    concurso: int,
    data_sorteio: str,
    real: Sequence[int],
    best: pd.Series,
    optimizer_summary: pd.DataFrame,
    generated_at: str,
    extra: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    nums = parse_numbers(str(best["nums"]))
    hits = sorted(set(nums) & set(int(n) for n in real))
    wrong = sorted(set(nums) - set(int(n) for n in real))
    missing = sorted(set(int(n) for n in real) - set(nums))
    row = {
        "modelo_nome": SOURCE_MODEL_SINGLE,
        "weight_profile": profile_name,
        "score_weights": format_exhaustive_weights(weights),
        "data_geracao_jogo": generated_at,
        "concurso_previsto": int(concurso),
        "data_sorteio": data_sorteio,
        "jogo_unico": format_nums(nums),
        "numeros_reais": format_nums(real),
        "qtd_acertos": int(len(hits)),
        "dezenas_acertadas": format_nums(hits) if hits else "",
        "dezenas_erradas": format_nums(wrong) if wrong else "",
        "dezenas_faltantes": format_nums(missing) if missing else "",
        "acertou_11": int(len(hits) >= 11),
        "acertou_12": int(len(hits) >= 12),
        "acertou_13": int(len(hits) >= 13),
        "acertou_14": int(len(hits) >= 14),
        "acertou_15": int(len(hits) >= 15),
        "score_final": float(best.get("score_final", 0.0)),
        "score_estatistico": float(best.get("score_estatistico", 0.0)),
        "score_historico": float(best.get("score_historico", 0.0)),
        "score_atraso": float(best.get("score_atraso", 0.0)),
        "score_combinatorio": float(best.get("score_combinatorio", 0.0)),
        "score_contextual": float(best.get("score_contextual", 0.0)),
        "score_localidade_numerologia": float(best.get("score_localidade_numerologia", 0.0)),
        "score_cenarios": float(best.get("score_cenarios", 0.0)),
        "score_contrarian": float(best.get("score_contrarian", 0.0)),
        "score_transicao": float(best.get("score_transicao", 0.0)),
        "total_combinacoes_avaliadas": int(_summary_value(optimizer_summary, "combinacoes_avaliadas", 0) or 0),
        "combinacoes_possiveis": int(_summary_value(optimizer_summary, "combinacoes_possiveis", TOTAL_COMBINATIONS) or TOTAL_COMBINATIONS),
        "historico_concursos_usados": int(_summary_value(optimizer_summary, "historico_concursos_usados", 0) or 0),
        "data_proximo_concurso_usada": _summary_value(optimizer_summary, "data_proximo_concurso", ""),
        "dia_semana_proximo_concurso": _summary_value(optimizer_summary, "dia_semana_proximo_concurso", ""),
        "fase_lua_proximo_concurso": _summary_value(optimizer_summary, "fase_lua_proximo_concurso", ""),
        "numerologia_data_raiz": _summary_value(optimizer_summary, "numerologia_data_raiz", ""),
        "local_sorteio_assumido": _summary_value(optimizer_summary, "local_sorteio_assumido", ""),
        "cidade_sorteio_assumida": _summary_value(optimizer_summary, "cidade_sorteio_assumida", ""),
        "uf_sorteio_assumida": _summary_value(optimizer_summary, "uf_sorteio_assumida", ""),
    }
    if extra:
        row.update(dict(extra))
    return row


def summarize_validation_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for profile_name, group in results.groupby("weight_profile"):
        hits = group["qtd_acertos"].astype(int)
        rows.append(
            {
                "weight_profile": profile_name,
                "n_concursos": int(len(group)),
                "media_acertos_jogo_unico": round(float(hits.mean()), 6),
                "mediana_acertos_jogo_unico": round(float(hits.median()), 6),
                "min_acertos": int(hits.min()),
                "max_acertos": int(hits.max()),
                "p_acertou_11": round(float((hits >= 11).mean()), 6),
                "p_acertou_12": round(float((hits >= 12).mean()), 6),
                "p_acertou_13": round(float((hits >= 13).mean()), 6),
                "p_acertou_14": round(float((hits >= 14).mean()), 6),
                "p_acertou_15": round(float((hits >= 15).mean()), 6),
                "total_combinacoes_avaliadas": int(group["total_combinacoes_avaliadas"].astype(int).sum()),
                "score_weights": str(group["score_weights"].iloc[0]),
            }
        )
    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["media_acertos_jogo_unico", "max_acertos", "weight_profile"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def run_exhaustive_single_backtest(
    concursos: pd.DataFrame,
    *,
    n_eval: int = 3,
    min_history: int = 300,
    top_games: int = 20,
    draw_hour: int = 20,
    draw_minute: int = 0,
    exhaustive_limit: int | None = None,
    weight_profile: str = "padrao_atual",
    weights: Mapping[str, float] | None = None,
    extra: Mapping[str, object] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if concursos.empty:
        raise ValueError("Base de concursos vazia. Rode primeiro: python main.py --update")
    if n_eval <= 0:
        raise ValueError("n_eval deve ser maior que zero.")
    if min_history < 10:
        raise ValueError("min_history deve ser pelo menos 10.")

    df = concursos.copy().sort_values("concurso").reset_index(drop=True)
    total = len(df)
    if total <= min_history:
        raise ValueError(f"Historico insuficiente: {total} concursos para min_history={min_history}.")

    resolved_weights = resolve_exhaustive_weights(weights if weights is not None else weights_for_profile(weight_profile))
    draws = [nums_from_row(row) for _, row in df.iterrows()]
    start_idx = max(int(min_history), total - int(n_eval))
    generated_at = datetime.now().isoformat(timespec="seconds")
    rows: List[Dict[str, object]] = []

    for idx in range(start_idx, total):
        train_df = df.iloc[:idx].copy()
        real = draws[idx]
        concurso = int(df.loc[idx, "concurso"])
        data_sorteio = str(df.loc[idx, "data_sorteio"])
        candidates, optimizer_summary = build_exhaustive_candidates(
            train_df,
            top_games=max(1, int(top_games)),
            draw_hour=draw_hour,
            draw_minute=draw_minute,
            limit_combinations=exhaustive_limit,
            weights=resolved_weights,
        )
        best = _best_candidate(candidates)
        rows.append(
            _result_row(
                profile_name=weight_profile,
                weights=resolved_weights,
                concurso=concurso,
                data_sorteio=data_sorteio,
                real=real,
                best=best,
                optimizer_summary=optimizer_summary,
                generated_at=generated_at,
                extra=extra,
            )
        )

    results = pd.DataFrame(rows)
    return results, summarize_validation_results(results)


def run_ablation_test(
    concursos: pd.DataFrame,
    *,
    n_eval: int = 3,
    min_history: int = 300,
    top_games: int = 20,
    draw_hour: int = 20,
    draw_minute: int = 0,
    exhaustive_limit: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base_weights = weights_for_profile("padrao_atual")
    result_frames: List[pd.DataFrame] = []
    for variant_name, disabled_component in ABLATION_VARIANTS:
        variant_weights = base_weights if disabled_component is None else _disable_component(base_weights, disabled_component)
        results, _summary = run_exhaustive_single_backtest(
            concursos,
            n_eval=n_eval,
            min_history=min_history,
            top_games=top_games,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
            exhaustive_limit=exhaustive_limit,
            weight_profile=variant_name,
            weights=variant_weights,
            extra={
                "ablation_removed_component": disabled_component or "",
                "ablation_base_profile": "padrao_atual",
            },
        )
        result_frames.append(results)

    all_results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    summary = summarize_validation_results(all_results)
    if not summary.empty:
        baseline_values = summary.loc[summary["weight_profile"] == "completo", "media_acertos_jogo_unico"]
        baseline = float(baseline_values.iloc[0]) if not baseline_values.empty else float(summary["media_acertos_jogo_unico"].max())
        summary["delta_media_vs_completo"] = summary["media_acertos_jogo_unico"].astype(float) - baseline
        summary["ranking_ablation"] = range(1, len(summary) + 1)
        summary = summary[
            ["ranking_ablation"]
            + [col for col in summary.columns if col != "ranking_ablation"]
        ]
    return all_results, summary


def run_weight_tuning(
    concursos: pd.DataFrame,
    *,
    n_eval: int = 3,
    min_history: int = 300,
    top_games: int = 20,
    draw_hour: int = 20,
    draw_minute: int = 0,
    exhaustive_limit: int | None = None,
    profiles: Sequence[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    selected_profiles = list(profiles or WEIGHT_PROFILE_PRESETS.keys())
    result_frames: List[pd.DataFrame] = []
    for profile_name in selected_profiles:
        profile_weights = weights_for_profile(profile_name)
        results, _summary = run_exhaustive_single_backtest(
            concursos,
            n_eval=n_eval,
            min_history=min_history,
            top_games=top_games,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
            exhaustive_limit=exhaustive_limit,
            weight_profile=profile_name,
            weights=profile_weights,
            extra={"tuning_profile": profile_name},
        )
        result_frames.append(results)

    all_results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    summary = summarize_validation_results(all_results)
    if not summary.empty:
        summary.insert(0, "ranking_profile", range(1, len(summary) + 1))
        best_profile = str(summary.loc[0, "weight_profile"])
        best_weights = weights_for_profile(best_profile)
    else:
        best_profile = ""
        best_weights = {}
    best_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "best_profile": best_profile,
        "best_weights": best_weights,
        "score_weights": format_exhaustive_weights(best_weights) if best_weights else "",
        "n_eval": int(n_eval),
        "min_history": int(min_history),
        "top_games": int(top_games),
        "exhaustive_limit": exhaustive_limit,
    }
    return all_results, summary, best_payload
