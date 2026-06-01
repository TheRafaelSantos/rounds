from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from .context_features import TargetContext, build_target_context
from .exhaustive_optimizer import EXHAUSTIVE_SOURCE_MODEL, TOTAL_COMBINATIONS, build_exhaustive_candidates
from .optimizer import build_optimized_candidates


AVISO_TECNICO = "Sugestoes matematicas/estatisticas; nao existe garantia de acerto em sorteios aleatorios."


@dataclass(frozen=True)
class PredictionSummary:
    concurso_alvo: int
    generated_at: str
    data_proximo_concurso: str
    dia_semana: str
    fase_lua: str
    iluminacao_lua_percentual: float
    numerologia_data_raiz: int
    jogo_1: str
    jogo_2: str
    prediction_csv_path: str
    report_path: str
    excel_path: str
    metodo: str = EXHAUSTIVE_SOURCE_MODEL

    def to_console(self) -> str:
        return "\n".join(
            [
                "",
                f"Concurso-alvo: {self.concurso_alvo}",
                f"Gerado em: {self.generated_at}",
                f"Data do proximo concurso: {self.data_proximo_concurso}",
                f"Dia da semana: {self.dia_semana}",
                f"Lua no horario de Brasilia: {self.fase_lua} ({self.iluminacao_lua_percentual:.2f}% iluminada)",
                f"Numerologia da data: raiz {self.numerologia_data_raiz}",
                f"Metodo: {self.metodo}",
                f"Jogo 1 principal: {self.jogo_1}",
                f"Jogo 2 alternativo completo: {self.jogo_2}",
                f"Aviso: {AVISO_TECNICO}",
                f"Relatorio tecnico: {self.report_path}",
            ]
        )


def _parse_nums(text: str) -> List[int]:
    nums = [int(part) for part in str(text).split()]
    if len(nums) != 15 or len(set(nums)) != 15 or any(n < 1 or n > 25 for n in nums):
        raise ValueError(f"Candidato invalido: {text}")
    return sorted(nums)


def _overlap(a: Sequence[int], b: Sequence[int]) -> int:
    return len(set(a) & set(b))


def _format_nums(nums: Sequence[int]) -> str:
    return " ".join(f"{int(n):02d}" for n in sorted(nums))


def _safe_float(row: pd.Series, column: str, default: float = 0.0) -> float:
    if column not in row:
        return default
    try:
        value = float(row.get(column, default))
    except (TypeError, ValueError):
        return default
    if pd.isna(value):
        return default
    return value


def _mean_available_scores(row: pd.Series, columns: Sequence[str]) -> float:
    values: List[float] = []
    for column in columns:
        if column not in row:
            continue
        try:
            value = float(row.get(column))
        except (TypeError, ValueError):
            continue
        if not pd.isna(value):
            values.append(value)
    if not values:
        return _safe_float(row, "score_final")
    return sum(values) / len(values)


def _portfolio_metrics_for_second(row: pd.Series, first_nums: Sequence[int]) -> Dict[str, object]:
    nums = _parse_nums(str(row["nums"]))
    overlap = _overlap(first_nums, nums)
    unique_second = sorted(set(nums) - set(first_nums))
    unique_first = sorted(set(first_nums) - set(nums))
    diversity_score = round(((15 - overlap) / 15) * 100, 6)
    strength_score = round(
        _mean_available_scores(
            row,
            [
                "score_estatistico",
                "score_historico",
                "score_atraso",
                "score_combinatorio",
                "score_cenarios",
                "score_contrarian",
            ],
        ),
        6,
    )
    final_score = _safe_float(row, "score_final")
    transition_score = _safe_float(row, "score_transicao", final_score)
    contextual_score = _safe_float(row, "score_contextual", _safe_float(row, "score_localidade_numerologia", final_score))
    portfolio_score = round(
        (final_score * 0.58)
        + (transition_score * 0.14)
        + (contextual_score * 0.10)
        + (diversity_score * 0.12)
        + (strength_score * 0.06),
        6,
    )
    return {
        "overlap_com_jogo_1": overlap,
        "dezenas_unicas_vs_jogo_1": 15 - overlap,
        "dezenas_exclusivas_jogo_2": _format_nums(unique_second),
        "dezenas_exclusivas_jogo_1": _format_nums(unique_first),
        "score_diversidade_jogo_2": diversity_score,
        "score_forca_componentes_jogo_2": strength_score,
        "score_portfolio_jogo_2": portfolio_score,
    }


def select_final_games(candidates: pd.DataFrame, *, max_overlap: int = 10) -> pd.DataFrame:
    if candidates.empty:
        raise ValueError("Nenhum candidato disponivel para selecao final.")
    if "nums" not in candidates.columns or "score_final" not in candidates.columns:
        raise ValueError("Tabela de candidatos precisa ter colunas nums e score_final.")

    ranked = candidates.copy().sort_values(["score_final", "nums"], ascending=[False, True]).reset_index(drop=True)
    first = ranked.iloc[0].copy()
    first_nums = _parse_nums(first["nums"])
    first["criterio_selecao"] = "principal_rank_1"
    first["overlap_com_jogo_1"] = 15
    first["dezenas_unicas_vs_jogo_1"] = 0
    first["dezenas_exclusivas_jogo_2"] = ""
    first["dezenas_exclusivas_jogo_1"] = ""
    first["score_diversidade_jogo_2"] = pd.NA
    first["score_forca_componentes_jogo_2"] = pd.NA
    first["score_portfolio_jogo_2"] = pd.NA

    eligible_second_rows: List[pd.Series] = []
    for _, row in ranked.iloc[1:].iterrows():
        nums = _parse_nums(row["nums"])
        if nums != first_nums and _overlap(first_nums, nums) <= int(max_overlap):
            enriched = row.copy()
            for key, value in _portfolio_metrics_for_second(enriched, first_nums).items():
                enriched[key] = value
            enriched["criterio_selecao"] = f"portfolio_inteligente_overlap<={int(max_overlap)}"
            eligible_second_rows.append(enriched)

    if eligible_second_rows:
        second_ranked = pd.DataFrame(eligible_second_rows).sort_values(
            ["score_portfolio_jogo_2", "score_final", "nums"],
            ascending=[False, False, True],
        )
        second = second_ranked.iloc[0].copy()
    else:
        second = None
    if second is None:
        for _, row in ranked.iloc[1:].iterrows():
            nums = _parse_nums(row["nums"])
            if nums != first_nums:
                second = row.copy()
                for key, value in _portfolio_metrics_for_second(second, first_nums).items():
                    second[key] = value
                second["criterio_selecao"] = "fallback_sem_overlap_minimo"
                break
    if second is None:
        raise ValueError("Nao foi possivel selecionar dois jogos distintos.")

    out = pd.DataFrame([first, second]).reset_index(drop=True)
    out.insert(0, "jogo", [1, 2])
    return out


def build_prediction_report(
    *,
    final_games: pd.DataFrame,
    concurso_alvo: int,
    generated_at: str,
    last_concurso: int,
    last_date: str,
    max_overlap: int,
    target_context: TargetContext,
    source_model: str,
) -> str:
    lines = [
        "# Relatorio tecnico - Lotofacil Analytics",
        "",
        f"Gerado em: {generated_at}",
        f"Ultimo concurso na base: {last_concurso} ({last_date})",
        f"Concurso-alvo estimado: {concurso_alvo}",
        f"Metodo: {source_model}",
        f"Diversidade maxima configurada entre jogos: overlap <= {max_overlap}",
        "",
        "## Contexto do proximo concurso",
        "",
        f"- Data do proximo concurso: {target_context.data_proximo_concurso}",
        f"- Horario de Brasilia usado no calculo lunar: {target_context.horario_brasilia_assumido}",
        f"- Dia da semana: {target_context.dia_semana_nome}",
        f"- Periodo do ano: {target_context.estacao_do_ano}; trimestre {target_context.trimestre}; semestre {target_context.semestre}",
        f"- Lua: {target_context.fase_lua}; idade {target_context.idade_lua:.2f} dias; iluminacao {target_context.iluminacao_lua_percentual:.2f}%",
        f"- Numerologia: raiz da data {target_context.numerologia_data_raiz}; raiz do concurso {target_context.numerologia_concurso_raiz}; raiz dia+mes {target_context.numerologia_dia_mes_raiz}",
        f"- Observacao: {target_context.observacao_horario}",
        f"- Localidade usada: {target_context.local_sorteio_assumido or '-'} | {target_context.cidade_sorteio_assumida or '-'} | {target_context.uf_sorteio_assumida or '-'}",
        f"- Bairro: {target_context.bairro_sorteio_assumido or 'indisponivel na base atual'}",
        f"- Observacao localidade: {target_context.observacao_localidade}",
        "",
        "Aviso: sugestoes matematicas/estatisticas; nao existe garantia de acerto em sorteios aleatorios.",
        "",
        "## Jogos finais",
        "",
    ]
    for _, row in final_games.iterrows():
        parts = [
            f"- Jogo {int(row['jogo'])}: {row['nums']}",
            f"score_final={_safe_float(row, 'score_final'):.6f}",
            f"score_contextual={_safe_float(row, 'score_contextual'):.6f}",
            f"score_transicao={_safe_float(row, 'score_transicao'):.6f}",
            f"overlap_com_jogo_1={int(_safe_float(row, 'overlap_com_jogo_1'))}",
            f"criterio={row.get('criterio_selecao', '-')}",
            f"metodo_origem={row.get('metodo', '-')}",
        ]
        if int(row["jogo"]) == 2:
            parts.extend(
                [
                    f"score_portfolio_jogo_2={_safe_float(row, 'score_portfolio_jogo_2'):.6f}",
                    f"score_diversidade_jogo_2={_safe_float(row, 'score_diversidade_jogo_2'):.6f}",
                    f"dezenas_exclusivas_jogo_2={row.get('dezenas_exclusivas_jogo_2', '')}",
                ]
            )
        lines.append(" | ".join(parts))
    lines.extend(
        [
            "",
            "## Criterios usados",
            "",
            "1. score estatistico de equilibrio;",
            "2. score historico recente;",
            "3. score de atraso historico;",
            "4. score combinatorio;",
            "5. score contextual: data do proximo concurso, dia da semana, periodo do ano, fase da lua, numerologia e localidade;",
            "6. score de cenarios: soma baixa/media/alta, sequencias, faixas historicas e visual forte permitido;",
            "7. score contrarian: protege dezenas que o score tradicional poderia excluir, como canto, centro e faixa alta;",
            "8. score de transicao: compara cada concurso N com N+1 no historico e pontua repeticoes, entradas, saidas e estrutura de mudanca;",
            "9. varredura exaustiva das combinacoes possiveis;",
            "10. seletor inteligente do Jogo 2: entre os candidatos que respeitam a diversidade minima, escolhe o melhor score de portfolio combinando score final, transicao, contexto, forca dos componentes e dezenas diferentes do Jogo 1;",
            "11. diversidade minima entre os dois jogos.",
            "",
            f"Combinacoes possiveis da Lotofacil: {TOTAL_COMBINATIONS}.",
            "Os dois jogos sao combinacoes completas de 15 dezenas. O sistema nao divide um palpite em metades entre sugestoes.",
            "",
            "## Limite tecnico",
            "",
            "O proprio backtest e o ML desta versao nao demonstraram superioridade consistente contra o baseline aleatorio. Portanto, estes jogos devem ser tratados como saida experimental do processo, nao como previsao garantida.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_final_prediction(
    concursos: pd.DataFrame,
    *,
    existing_candidates: pd.DataFrame | None,
    seed: int,
    candidate_pool: int,
    top_games: int,
    generations: int,
    population: int,
    max_overlap: int,
    draw_hour: int,
    draw_minute: int,
    prediction_csv_path: Path,
    report_path: Path,
    excel_path: Path,
    engine: str = "exaustivo",
    exhaustive_limit: int | None = None,
) -> PredictionSummary:
    if concursos.empty:
        raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")

    df = concursos.copy().sort_values("concurso").reset_index(drop=True)
    last = df.iloc[-1]
    last_concurso = int(last["concurso"])
    last_date = str(last["data_sorteio"])
    concurso_alvo = last_concurso + 1
    target_context = build_target_context(df, draw_hour=draw_hour, draw_minute=draw_minute)

    source_model = EXHAUSTIVE_SOURCE_MODEL if engine == "exaustivo" else "ensemble_score_v2"
    required_context_cols = {"score_contextual", "score_transicao", "contexto_data_proximo_concurso", "contexto_fase_lua"}
    has_full_exhaustive_scan = True
    if engine == "exaustivo":
        if existing_candidates is not None and not existing_candidates.empty and "total_combinacoes_avaliadas" in existing_candidates.columns:
            max_evaluated = pd.to_numeric(existing_candidates["total_combinacoes_avaliadas"], errors="coerce").max()
            has_full_exhaustive_scan = bool(pd.notna(max_evaluated) and int(max_evaluated) >= TOTAL_COMBINATIONS)
        else:
            has_full_exhaustive_scan = False
    has_matching_engine = (
        existing_candidates is not None
        and not existing_candidates.empty
        and required_context_cols.issubset(existing_candidates.columns)
        and has_full_exhaustive_scan
        and (
            (engine != "exaustivo" and "source_model" not in existing_candidates.columns)
            or str(existing_candidates["source_model"].iloc[0]) == source_model
        )
    )
    if has_matching_engine:
        candidates = existing_candidates.copy()
    elif engine == "exaustivo":
        candidates, _summary = build_exhaustive_candidates(
            df,
            top_games=max(top_games, 5000),
            draw_hour=draw_hour,
            draw_minute=draw_minute,
            limit_combinations=exhaustive_limit,
        )
    else:
        candidates, _summary = build_optimized_candidates(
            df,
            seed=seed,
            candidate_pool=candidate_pool,
            top_games=max(top_games, 50),
            generations=generations,
            population=population,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
        )

    final_games = select_final_games(candidates, max_overlap=max_overlap)
    generated_at = datetime.now().isoformat(timespec="seconds")
    final_games.insert(0, "generated_at", generated_at)
    final_games.insert(1, "concurso_alvo", concurso_alvo)
    final_games.insert(2, "data_proximo_concurso", target_context.data_proximo_concurso)
    final_games.insert(3, "horario_brasilia_assumido", target_context.horario_brasilia_assumido)
    final_games.insert(4, "dia_semana_proximo_concurso", target_context.dia_semana_nome)
    final_games.insert(5, "fase_lua_proximo_concurso", target_context.fase_lua)
    final_games.insert(6, "iluminacao_lua_percentual", target_context.iluminacao_lua_percentual)
    final_games.insert(7, "numerologia_data_raiz", target_context.numerologia_data_raiz)
    if "source_model" in final_games.columns:
        final_games["source_model"] = source_model
    else:
        final_games.insert(8, "source_model", source_model)
    final_games["aviso"] = AVISO_TECNICO

    prediction_csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    final_games.to_csv(prediction_csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        final_games.to_excel(writer, index=False, sheet_name="jogos_finais")
    report_path.write_text(
        build_prediction_report(
            final_games=final_games,
            concurso_alvo=concurso_alvo,
            generated_at=generated_at,
            last_concurso=last_concurso,
            last_date=last_date,
            max_overlap=max_overlap,
            target_context=target_context,
            source_model=source_model,
        ),
        encoding="utf-8",
    )

    return PredictionSummary(
        concurso_alvo=concurso_alvo,
        generated_at=generated_at,
        data_proximo_concurso=target_context.data_proximo_concurso,
        dia_semana=target_context.dia_semana_nome,
        fase_lua=target_context.fase_lua,
        iluminacao_lua_percentual=target_context.iluminacao_lua_percentual,
        numerologia_data_raiz=target_context.numerologia_data_raiz,
        jogo_1=str(final_games.loc[0, "nums"]),
        jogo_2=str(final_games.loc[1, "nums"]),
        prediction_csv_path=str(prediction_csv_path),
        report_path=str(report_path),
        excel_path=str(excel_path),
        metodo=source_model,
    )
