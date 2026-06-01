from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from .context_features import TargetContext, build_target_context
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
                "Metodo: ensemble_score_v2",
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


def select_final_games(candidates: pd.DataFrame, *, max_overlap: int = 10) -> pd.DataFrame:
    if candidates.empty:
        raise ValueError("Nenhum candidato disponivel para selecao final.")
    if "nums" not in candidates.columns or "score_final" not in candidates.columns:
        raise ValueError("Tabela de candidatos precisa ter colunas nums e score_final.")

    ranked = candidates.copy().sort_values(["score_final", "nums"], ascending=[False, True]).reset_index(drop=True)
    first = ranked.iloc[0].copy()
    first_nums = _parse_nums(first["nums"])
    second = None
    for _, row in ranked.iloc[1:].iterrows():
        nums = _parse_nums(row["nums"])
        if nums != first_nums and _overlap(first_nums, nums) <= int(max_overlap):
            second = row.copy()
            break
    if second is None:
        for _, row in ranked.iloc[1:].iterrows():
            nums = _parse_nums(row["nums"])
            if nums != first_nums:
                second = row.copy()
                break
    if second is None:
        raise ValueError("Nao foi possivel selecionar dois jogos distintos.")

    out = pd.DataFrame([first, second]).reset_index(drop=True)
    out.insert(0, "jogo", [1, 2])
    out["overlap_com_jogo_1"] = [15, _overlap(first_nums, _parse_nums(str(second["nums"])))]
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
) -> str:
    lines = [
        "# Relatorio tecnico - Lotofacil Analytics",
        "",
        f"Gerado em: {generated_at}",
        f"Ultimo concurso na base: {last_concurso} ({last_date})",
        f"Concurso-alvo estimado: {concurso_alvo}",
        "Metodo: ensemble_score_v2",
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
        "",
        "Aviso: sugestoes matematicas/estatisticas; nao existe garantia de acerto em sorteios aleatorios.",
        "",
        "## Jogos finais",
        "",
    ]
    for _, row in final_games.iterrows():
        lines.append(
            f"- Jogo {int(row['jogo'])}: {row['nums']} | "
            f"score_final={float(row['score_final']):.6f} | "
            f"score_contextual={float(row.get('score_contextual', 0)):.6f} | "
            f"metodo_origem={row.get('metodo', '-')}"
        )
    lines.extend(
        [
            "",
            "## Criterios usados",
            "",
            "1. score estatistico de equilibrio;",
            "2. score historico recente;",
            "3. score anti-popularidade humana;",
            "4. score combinatorio;",
            "5. score contextual: data do proximo concurso, dia da semana, periodo do ano, fase da lua e numerologia exploratoria;",
            "6. score de cenarios: soma baixa/media/alta, sequencias, faixas historicas e visual forte permitido;",
            "7. score contrarian: protege dezenas que o score tradicional poderia excluir, como canto, centro e faixa alta;",
            "8. diversidade minima entre os dois jogos.",
            "",
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
) -> PredictionSummary:
    if concursos.empty:
        raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")

    df = concursos.copy().sort_values("concurso").reset_index(drop=True)
    last = df.iloc[-1]
    last_concurso = int(last["concurso"])
    last_date = str(last["data_sorteio"])
    concurso_alvo = last_concurso + 1
    target_context = build_target_context(df, draw_hour=draw_hour, draw_minute=draw_minute)

    required_context_cols = {"score_contextual", "contexto_data_proximo_concurso", "contexto_fase_lua"}
    if existing_candidates is not None and not existing_candidates.empty and required_context_cols.issubset(existing_candidates.columns):
        candidates = existing_candidates.copy()
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
    )
