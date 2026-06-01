from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class PostResultSummary:
    label: str
    concurso: int | None
    game_rows: int
    dezena_rows: int
    best_hits: int
    union_hits: int
    games_csv_path: str
    dezenas_csv_path: str
    report_path: str
    excel_path: str

    def to_console(self) -> str:
        return "\n".join(
            [
                "",
                "Resumo Lotofacil Analytics - Pos-sorteio",
                f"Rotulo: {self.label}",
                f"Concurso: {self.concurso if self.concurso is not None else '-'}",
                f"Jogos analisados: {self.game_rows}",
                f"Melhor jogo: {self.best_hits} acertos",
                f"Cobertura uniao dos jogos: {self.union_hits} dezenas",
                f"CSV jogos: {self.games_csv_path}",
                f"CSV dezenas: {self.dezenas_csv_path}",
                f"Relatorio: {self.report_path}",
                f"Excel: {self.excel_path}",
                "Mensagem: Analise pos-sorteio e auditoria de dezenas geradas.",
            ]
        )


def parse_numbers(text: str) -> List[int]:
    nums = [int(part) for part in re.findall(r"\d+", str(text))]
    if len(nums) != 15:
        raise ValueError(f"Esperado exatamente 15 dezenas, recebido {len(nums)}: {nums}")
    if len(set(nums)) != 15:
        raise ValueError(f"Dezenas repetidas no conjunto: {nums}")
    invalid = [n for n in nums if n < 1 or n > 25]
    if invalid:
        raise ValueError(f"Dezenas fora de 1..25: {invalid}")
    return sorted(nums)


def format_nums(nums: Sequence[int]) -> str:
    return " ".join(f"{int(n):02d}" for n in sorted(nums))


def _load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(f"Arquivo de predicao nao encontrado: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "nums" not in df.columns:
        raise ValueError("Arquivo de predicao precisa ter coluna nums.")
    if "jogo" not in df.columns:
        df = df.copy()
        df.insert(0, "jogo", range(1, len(df) + 1))
    return df


def _optimizer_dezena_view(optimizer_candidates: pd.DataFrame, *, top_n: int = 100) -> Dict[int, Dict[str, object]]:
    stats: Dict[int, Dict[str, object]] = {
        dezena: {
            "optimizer_top_ocorrencias": 0,
            "optimizer_melhor_rank": None,
            "optimizer_media_score_final": None,
            "optimizer_media_score_contextual": None,
        }
        for dezena in range(1, 26)
    }
    if optimizer_candidates.empty or "nums" not in optimizer_candidates.columns:
        return stats
    df = optimizer_candidates.copy().head(max(1, int(top_n)))
    score_values: Dict[int, List[float]] = {dezena: [] for dezena in range(1, 26)}
    context_values: Dict[int, List[float]] = {dezena: [] for dezena in range(1, 26)}
    for _, row in df.iterrows():
        nums = parse_numbers(str(row["nums"]))
        rank = int(row.get("rank", len(df)))
        for dezena in nums:
            stats[dezena]["optimizer_top_ocorrencias"] = int(stats[dezena]["optimizer_top_ocorrencias"]) + 1
            current_rank = stats[dezena]["optimizer_melhor_rank"]
            if current_rank is None or rank < int(current_rank):
                stats[dezena]["optimizer_melhor_rank"] = rank
            if "score_final" in row and not pd.isna(row["score_final"]):
                score_values[dezena].append(float(row["score_final"]))
            if "score_contextual" in row and not pd.isna(row["score_contextual"]):
                context_values[dezena].append(float(row["score_contextual"]))
    for dezena in range(1, 26):
        if score_values[dezena]:
            stats[dezena]["optimizer_media_score_final"] = round(float(pd.Series(score_values[dezena]).mean()), 6)
        if context_values[dezena]:
            stats[dezena]["optimizer_media_score_contextual"] = round(float(pd.Series(context_values[dezena]).mean()), 6)
    return stats


def _game_rows(predictions: pd.DataFrame, actual: Sequence[int]) -> pd.DataFrame:
    actual_set = set(actual)
    rows: List[Dict[str, object]] = []
    for _, row in predictions.iterrows():
        nums = parse_numbers(str(row["nums"]))
        nums_set = set(nums)
        hits = sorted(nums_set & actual_set)
        wrong = sorted(nums_set - actual_set)
        missing = sorted(actual_set - nums_set)
        rows.append(
            {
                "jogo": int(row["jogo"]),
                "nums": format_nums(nums),
                "qtd_acertos": len(hits),
                "acertos": format_nums(hits),
                "erros_no_jogo": format_nums(wrong),
                "dezenas_sorteadas_faltantes": format_nums(missing),
                "qtd_erros_no_jogo": len(wrong),
                "qtd_faltantes": len(missing),
                "troca_necessaria_para_15": f"remover {format_nums(wrong)} | incluir {format_nums(missing)}",
                "score_final": row.get("score_final"),
                "score_contextual": row.get("score_contextual"),
                "metodo": row.get("metodo", row.get("metodo_geracao", "-")),
            }
        )
    return pd.DataFrame(rows)


def _dezena_rows(
    predictions: pd.DataFrame,
    actual: Sequence[int],
    optimizer_candidates: pd.DataFrame,
) -> pd.DataFrame:
    actual_set = set(actual)
    games = {int(row["jogo"]): set(parse_numbers(str(row["nums"]))) for _, row in predictions.iterrows()}
    union_games = set().union(*games.values()) if games else set()
    optimizer_stats = _optimizer_dezena_view(optimizer_candidates)
    rows: List[Dict[str, object]] = []
    for dezena in range(1, 26):
        in_actual = dezena in actual_set
        in_any = dezena in union_games
        in_game_1 = dezena in games.get(1, set())
        in_game_2 = dezena in games.get(2, set())
        if in_actual and not in_any:
            classificacao = "falso_negativo_total"
        elif in_actual and not in_game_1:
            classificacao = "faltou_no_jogo_1"
        elif (not in_actual) and in_any:
            classificacao = "falso_positivo"
        elif in_actual:
            classificacao = "acerto"
        else:
            classificacao = "corretamente_fora"
        reason = []
        if dezena == 1:
            reason.append("dezena_canto_baixa_monitorar_penalizacao_visual")
        if dezena == 13:
            reason.append("dezena_central_monitorar_penalizacao_visual")
        if dezena in {21, 22, 24, 25}:
            reason.append("dezena_faixa_alta_monitorar_cobertura_21_25")
        if dezena in {5, 6, 7, 8, 9, 10}:
            reason.append("dezena_em_bloco_sequencial_monitorar_penalizacao_sequencia")
        rows.append(
            {
                "dezena": dezena,
                "saiu_no_resultado": int(in_actual),
                "entrou_jogo_1": int(in_game_1),
                "entrou_jogo_2": int(in_game_2),
                "entrou_em_algum_jogo": int(in_any),
                "classificacao": classificacao,
                "hipotese_auditoria": ";".join(reason) if reason else "",
                **optimizer_stats[dezena],
            }
        )
    return pd.DataFrame(rows)


def _report_text(
    *,
    label: str,
    concurso: int | None,
    actual: Sequence[int],
    game_results: pd.DataFrame,
    dezena_audit: pd.DataFrame,
) -> str:
    best_hits = int(game_results["qtd_acertos"].max()) if len(game_results) else 0
    union_hits = int(dezena_audit[(dezena_audit["saiu_no_resultado"] == 1) & (dezena_audit["entrou_em_algum_jogo"] == 1)].shape[0])
    missed_all = dezena_audit[(dezena_audit["saiu_no_resultado"] == 1) & (dezena_audit["entrou_em_algum_jogo"] == 0)]["dezena"].tolist()
    lines = [
        "# Analise pos-sorteio - Lotofacil Analytics",
        "",
        f"Gerado em: {datetime.now().isoformat(timespec='seconds')}",
        f"Rotulo: {label}",
        f"Concurso: {concurso if concurso is not None else '-'}",
        f"Resultado real: {format_nums(actual)}",
        "",
        "## Resumo",
        "",
        f"- Melhor jogo: {best_hits} acertos",
        f"- Cobertura dos jogos combinados: {union_hits} de 15 dezenas",
        f"- Dezenas sorteadas fora de todos os jogos: {format_nums(missed_all) if missed_all else '-'}",
        "",
        "## Jogos",
        "",
    ]
    for _, row in game_results.iterrows():
        lines.extend(
            [
                f"### Jogo {int(row['jogo'])}",
                "",
                f"- Acertos: {int(row['qtd_acertos'])}",
                f"- Dezenas acertadas: {row['acertos']}",
                f"- Erros dentro do jogo: {row['erros_no_jogo']}",
                f"- Dezenas que faltaram para 15: {row['dezenas_sorteadas_faltantes']}",
                f"- Troca necessaria: {row['troca_necessaria_para_15']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Hipoteses tecnicas",
            "",
            "1. Se dezenas corretas ficaram fora dos dois jogos, houve falso negativo total e o score precisa auditar por que elas foram descartadas.",
            "2. Se o jogo teve muitos erros internos, a selecao final precisa reduzir excesso de confianca no nucleo comum.",
            "3. Sequencias, cantos, centro do volante e faixa 21-25 precisam ser tratados como cenarios permitidos, nao como filtros rigidos.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze_post_result(
    *,
    actual_numbers: str,
    predictions_path: Path,
    optimizer_candidates_path: Path,
    games_csv_path: Path,
    dezenas_csv_path: Path,
    report_path: Path,
    excel_path: Path,
    label: str,
    concurso: int | None,
) -> PostResultSummary:
    actual = parse_numbers(actual_numbers)
    predictions = _load_predictions(predictions_path)
    optimizer_candidates = pd.read_csv(optimizer_candidates_path, encoding="utf-8-sig") if optimizer_candidates_path.exists() else pd.DataFrame()
    game_results = _game_rows(predictions, actual)
    dezena_audit = _dezena_rows(predictions, actual, optimizer_candidates)

    games_csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    game_results.to_csv(games_csv_path, index=False, encoding="utf-8-sig")
    dezena_audit.to_csv(dezenas_csv_path, index=False, encoding="utf-8-sig")
    report_path.write_text(
        _report_text(label=label, concurso=concurso, actual=actual, game_results=game_results, dezena_audit=dezena_audit),
        encoding="utf-8",
    )
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        game_results.to_excel(writer, index=False, sheet_name="jogos")
        dezena_audit.to_excel(writer, index=False, sheet_name="dezenas")

    best_hits = int(game_results["qtd_acertos"].max()) if len(game_results) else 0
    union_hits = int(dezena_audit[(dezena_audit["saiu_no_resultado"] == 1) & (dezena_audit["entrou_em_algum_jogo"] == 1)].shape[0])
    return PostResultSummary(
        label=label,
        concurso=concurso,
        game_rows=int(len(game_results)),
        dezena_rows=int(len(dezena_audit)),
        best_hits=best_hits,
        union_hits=union_hits,
        games_csv_path=str(games_csv_path),
        dezenas_csv_path=str(dezenas_csv_path),
        report_path=str(report_path),
        excel_path=str(excel_path),
    )
