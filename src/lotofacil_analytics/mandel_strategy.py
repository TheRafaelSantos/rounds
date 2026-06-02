from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from .context_features import build_target_context
from .exhaustive_optimizer import TOTAL_COMBINATIONS
from .selection_guard import (
    build_number_guard_table,
    enrich_candidates_with_decision_guard,
    format_nums,
    parse_nums,
    select_guarded_best_candidate,
)


TICKET_PRICE = 3.50
MIN_UNIVERSE_SIZE = 15
MAX_CAIXA_UNIVERSE_SIZE = 20
PICK_SIZE = 15
SOURCE_MODEL_MANDEL = "mandel_desdobramento_condensado_lotofacil"


@dataclass(frozen=True)
class MandelSummary:
    concurso_alvo: int
    generated_at: str
    data_proximo_concurso: str
    universo_recomendado: str
    tamanho_universo: int
    jogos_desdobramento_completo: int
    custo_desdobramento_completo: float
    jogos_fechamento_reduzido: int
    custo_fechamento_reduzido: float
    garantia_fechamento_reduzido: str
    plan_csv_path: str
    games_csv_path: str
    report_path: str
    excel_path: str

    def to_console(self) -> str:
        return "\n".join(
            [
                "",
                "Resumo Lotofacil Analytics - Mandel / Bolao",
                f"Concurso-alvo: {self.concurso_alvo}",
                f"Data do proximo concurso: {self.data_proximo_concurso}",
                f"Universo recomendado: {self.universo_recomendado}",
                f"Tamanho do universo: {self.tamanho_universo}",
                f"Desdobramento completo: {self.jogos_desdobramento_completo} jogos | {_money_ptbr(self.custo_desdobramento_completo)}",
                f"Fechamento reduzido: {self.jogos_fechamento_reduzido} jogos | {_money_ptbr(self.custo_fechamento_reduzido)}",
                f"Garantia do fechamento reduzido: {self.garantia_fechamento_reduzido}",
                f"CSV plano: {self.plan_csv_path}",
                f"CSV jogos: {self.games_csv_path}",
                f"Relatorio: {self.report_path}",
                f"Excel: {self.excel_path}",
                "Aviso: cobertura matematica aumenta chance por quantidade de jogos, mas nao cria previsao garantida do sorteio.",
            ]
        )


def _cost(games: int) -> float:
    return round(float(games) * TICKET_PRICE, 2)


def _money_ptbr(value: float) -> str:
    formatted = f"{float(value):,.2f}"
    return "R$ " + formatted.replace(",", "_").replace(".", ",").replace("_", ".")


def _number_ptbr(value: float, *, decimals: int = 2) -> str:
    formatted = f"{float(value):,.{int(decimals)}f}"
    return formatted.replace(",", "_").replace(".", ",").replace("_", ".")


def _coverage_probability(games: int) -> float:
    return round(float(games) / TOTAL_COMBINATIONS, 10)


def _odds_denominator(games: int) -> float:
    if games <= 0:
        return float("inf")
    return round(TOTAL_COMBINATIONS / float(games), 6)


def _candidate_number_scores(candidates: pd.DataFrame) -> Dict[int, float]:
    guard_table = build_number_guard_table(candidates)
    scores: Dict[int, float] = {}
    for _, row in guard_table.iterrows():
        scores[int(row["dezena"])] = float(row["score_protecao_falso_negativo"])
    return scores


def choose_strategy_universe(candidates: pd.DataFrame, *, universe_size: int) -> List[int]:
    if candidates.empty:
        raise ValueError("Nenhum candidato disponivel para montar universo Mandel.")
    size = max(MIN_UNIVERSE_SIZE, min(MAX_CAIXA_UNIVERSE_SIZE, int(universe_size)))
    best = select_guarded_best_candidate(candidates)
    selected = parse_nums(str(best["nums"]))
    score_map = _candidate_number_scores(candidates)
    for dezena, _score in sorted(score_map.items(), key=lambda item: (-item[1], item[0])):
        if len(selected) >= size:
            break
        if int(dezena) not in selected:
            selected.append(int(dezena))
    if len(selected) > size:
        selected = sorted(selected, key=lambda n: (-score_map.get(n, 0.0), n))[:size]
    return sorted(selected)


def build_plan_table(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    full_pot_cost = _cost(TOTAL_COMBINATIONS)
    for size in range(MIN_UNIVERSE_SIZE, MAX_CAIXA_UNIVERSE_SIZE + 1):
        games = math.comb(size, PICK_SIZE)
        universe = choose_strategy_universe(candidates, universe_size=size)
        rows.append(
            {
                "tamanho_universo": size,
                "universo_recomendado": format_nums(universe),
                "jogos_desdobramento_completo": int(games),
                "custo_desdobramento_completo": _cost(games),
                "chance_15_acertos_aproximada": _coverage_probability(games),
                "chance_15_acertos_1_em": _odds_denominator(games),
                "condicao_garantia_15": f"garante 15 somente se os 15 sorteados estiverem dentro das {size} dezenas",
                "custo_compra_todas_combinacoes": full_pot_cost,
                "metodo": SOURCE_MODEL_MANDEL,
            }
        )
    return pd.DataFrame(rows)


def _positions_mask(positions: Sequence[int]) -> int:
    mask = 0
    for position in positions:
        mask |= 1 << int(position)
    return mask


def _target_indexes_covered(
    block_mask: int,
    target_index: Dict[int, int],
    *,
    total_numbers: int,
    max_swaps: int,
) -> set[int]:
    selected = [idx for idx in range(total_numbers) if block_mask & (1 << idx)]
    omitted = [idx for idx in range(total_numbers) if not block_mask & (1 << idx)]
    covered: set[int] = set()
    max_allowed_swaps = min(int(max_swaps), len(selected), len(omitted))
    for swap_count in range(max_allowed_swaps + 1):
        for removed in combinations(selected, swap_count):
            base_mask = block_mask & ~_positions_mask(removed)
            for added in combinations(omitted, swap_count):
                target_mask = base_mask | _positions_mask(added)
                target_idx = target_index.get(target_mask)
                if target_idx is not None:
                    covered.add(target_idx)
    return covered


def greedy_reduced_closure(
    universe: Sequence[int],
    *,
    guarantee_hits: int = 14,
    max_games: int = 80,
) -> Tuple[List[Tuple[int, ...]], float, bool]:
    universe = tuple(sorted(int(n) for n in universe))
    if len(universe) < PICK_SIZE:
        raise ValueError("Universo precisa ter pelo menos 15 dezenas.")
    if not (11 <= int(guarantee_hits) <= PICK_SIZE):
        raise ValueError("Garantia precisa estar entre 11 e 15 acertos.")

    number_positions = {number: idx for idx, number in enumerate(universe)}
    blocks = list(combinations(universe, PICK_SIZE))
    block_masks = [_positions_mask([number_positions[number] for number in block]) for block in blocks]
    target_index = {mask: idx for idx, mask in enumerate(block_masks)}
    uncovered = set(range(len(blocks)))
    selected: List[Tuple[int, ...]] = []
    coverage_cache: Dict[int, set[int]] = {}
    max_swaps = PICK_SIZE - int(guarantee_hits)

    while uncovered and len(selected) < max(1, int(max_games)):
        best_block: Tuple[int, ...] | None = None
        best_cover: set[int] = set()
        for block_idx, block in enumerate(blocks):
            if block_idx in coverage_cache:
                covered = coverage_cache[block_idx] & uncovered
            else:
                coverage_cache[block_idx] = _target_indexes_covered(
                    block_masks[block_idx],
                    target_index,
                    total_numbers=len(universe),
                    max_swaps=max_swaps,
                )
                covered = coverage_cache[block_idx] & uncovered
            if len(covered) > len(best_cover) or (len(covered) == len(best_cover) and best_block is not None and block < best_block):
                best_block = block
                best_cover = covered
        if best_block is None or not best_cover:
            break
        selected.append(best_block)
        uncovered -= best_cover

    covered_count = len(blocks) - len(uncovered)
    coverage_pct = round(covered_count / len(blocks) * 100.0, 6) if blocks else 0.0
    return selected, coverage_pct, not uncovered


def build_games_table(
    *,
    universe: Sequence[int],
    guarantee_hits: int,
    max_reduced_games: int,
) -> Tuple[pd.DataFrame, float, bool]:
    games, coverage_pct, complete = greedy_reduced_closure(
        universe,
        guarantee_hits=guarantee_hits,
        max_games=max_reduced_games,
    )
    rows = [
        {
            "tipo": "fechamento_reduzido",
            "jogo": idx,
            "nums": format_nums(game),
            "tamanho_universo": len(universe),
            "universo": format_nums(universe),
            "garantia_alvo": int(guarantee_hits),
            "cobertura_condicional_pct": coverage_pct,
            "garantia_condicional_completa": int(complete),
            "metodo": SOURCE_MODEL_MANDEL,
        }
        for idx, game in enumerate(games, start=1)
    ]
    return pd.DataFrame(rows), coverage_pct, complete


def build_mandel_report(
    *,
    plan: pd.DataFrame,
    games: pd.DataFrame,
    concurso_alvo: int,
    data_proximo_concurso: str,
    generated_at: str,
    universe: Sequence[int],
    guarantee_hits: int,
    coverage_pct: float,
    complete: bool,
) -> str:
    recommended = plan.loc[plan["tamanho_universo"] == len(universe)].iloc[0]
    lines = [
        "# Relatorio tecnico - Mandel / Bolao Lotofacil",
        "",
        f"Gerado em: {generated_at}",
        f"Concurso-alvo estimado: {concurso_alvo}",
        f"Data do proximo concurso: {data_proximo_concurso}",
        "",
        "## O que foi aplicado",
        "",
        "1. Compra total do pote: calculada como referencia, mas tratada como inviavel pelo custo.",
        "2. Condensacao combinatoria: aplicada como universo recomendado de 15 a 20 dezenas.",
        "3. Desdobramento completo: todos os jogos de 15 dezenas dentro do universo escolhido.",
        "4. Fechamento reduzido: selecao gulosa de jogos para cobrir o maximo possivel dos cenarios de 15 dezenas dentro do universo.",
        "5. O motor atual continua escolhendo dezenas; o plano Mandel organiza cobertura e custo para bolao.",
        "",
        "## Universo recomendado",
        "",
        f"- Universo: {format_nums(universe)}",
        f"- Tamanho: {len(universe)} dezenas",
        f"- Desdobramento completo: {int(recommended['jogos_desdobramento_completo'])} jogos",
        f"- Custo do desdobramento completo: {_money_ptbr(float(recommended['custo_desdobramento_completo']))}",
        f"- Fechamento reduzido gerado: {len(games)} jogos",
        f"- Custo do fechamento reduzido: {_money_ptbr(_cost(len(games)))}",
        f"- Garantia alvo do fechamento: {guarantee_hits} pontos, condicionada a os 15 sorteados estarem dentro do universo",
        f"- Cobertura condicional do fechamento: {_number_ptbr(coverage_pct)}%",
        f"- Garantia condicional completa: {'sim' if complete else 'nao'}",
        "",
        "## Limite tecnico",
        "",
        "O metodo aumenta cobertura por quantidade de apostas. Ele nao aumenta a probabilidade individual de cada combinacao e nao garante lucro, porque ha custo, divisao de premio, impostos/regras e possibilidade de outros ganhadores.",
        "",
        "## Plano por tamanho de universo",
        "",
    ]
    for _, row in plan.iterrows():
        lines.append(
            f"- {int(row['tamanho_universo'])} dezenas: {int(row['jogos_desdobramento_completo'])} jogos | "
            f"{_money_ptbr(float(row['custo_desdobramento_completo']))} | "
            f"chance aproximada 1 em {_number_ptbr(float(row['chance_15_acertos_1_em']))}"
        )
    return "\n".join(lines) + "\n"


def run_mandel_strategy(
    concursos: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    universe_size: int,
    guarantee_hits: int,
    max_reduced_games: int,
    plan_csv_path: Path,
    games_csv_path: Path,
    report_path: Path,
    excel_path: Path,
    draw_hour: int = 20,
    draw_minute: int = 0,
) -> MandelSummary:
    if concursos.empty:
        raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")
    if candidates.empty:
        raise ValueError("Candidatos do otimizador nao encontrados. Rode primeiro: python main.py --optimize --engine exaustivo --top-games 5000")

    df = concursos.copy().sort_values("concurso").reset_index(drop=True)
    target_context = build_target_context(df, draw_hour=draw_hour, draw_minute=draw_minute)
    candidates_enriched = enrich_candidates_with_decision_guard(candidates)
    generated_at = pd.Timestamp.now().isoformat(timespec="seconds")
    universe = choose_strategy_universe(candidates_enriched, universe_size=universe_size)
    plan = build_plan_table(candidates_enriched)
    games, coverage_pct, complete = build_games_table(
        universe=universe,
        guarantee_hits=guarantee_hits,
        max_reduced_games=max_reduced_games,
    )

    plan_csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(plan_csv_path, index=False, encoding="utf-8-sig")
    games.to_csv(games_csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        plan.to_excel(writer, index=False, sheet_name="plano")
        games.to_excel(writer, index=False, sheet_name="fechamento_reduzido")
        build_number_guard_table(candidates_enriched).to_excel(writer, index=False, sheet_name="dezenas_guarda")

    report_path.write_text(
        build_mandel_report(
            plan=plan,
            games=games,
            concurso_alvo=target_context.concurso_alvo,
            data_proximo_concurso=target_context.data_proximo_concurso,
            generated_at=generated_at,
            universe=universe,
            guarantee_hits=guarantee_hits,
            coverage_pct=coverage_pct,
            complete=complete,
        ),
        encoding="utf-8",
    )

    full_games = math.comb(len(universe), PICK_SIZE)
    return MandelSummary(
        concurso_alvo=target_context.concurso_alvo,
        generated_at=generated_at,
        data_proximo_concurso=target_context.data_proximo_concurso,
        universo_recomendado=format_nums(universe),
        tamanho_universo=len(universe),
        jogos_desdobramento_completo=int(full_games),
        custo_desdobramento_completo=_cost(full_games),
        jogos_fechamento_reduzido=int(len(games)),
        custo_fechamento_reduzido=_cost(len(games)),
        garantia_fechamento_reduzido=f"{guarantee_hits} pontos condicionais; cobertura {_number_ptbr(coverage_pct)}%; completa={'sim' if complete else 'nao'}",
        plan_csv_path=str(plan_csv_path),
        games_csv_path=str(games_csv_path),
        report_path=str(report_path),
        excel_path=str(excel_path),
    )
