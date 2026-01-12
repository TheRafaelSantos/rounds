# analysis_transition.py
# Modelo de transição por POSIÇÃO (p1..p6) usando deltas entre concursos consecutivos.
# p1 = menor dezena do concurso, p6 = maior.
#
# A ideia: estimar P(delta | posição) com suavização, e pontuar um candidato
# pela probabilidade dos deltas vs o último concurso.

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd


MAX_N = 60
PICK = 6


def _sorted_nums(row_nums: Sequence[int]) -> Tuple[int, ...]:
    return tuple(sorted(int(x) for x in row_nums))


def build_transition_model(
    df: pd.DataFrame,
    *,
    recent_draws: int = 0,
    alpha: float = 0.5,
) -> Dict[str, object]:
    """
    Retorna um dict com:
      - probs_by_pos: lista len=6 de dict(delta->prob)
      - support: lista de deltas possíveis observados (por posição)
      - meta
    """
    if "nums" not in df.columns:
        raise ValueError("df precisa ter coluna nums (lista das dezenas).")

    arr = np.array([_sorted_nums(x) for x in df["nums"].tolist()], dtype=int)  # (n,6)
    if arr.shape[1] != PICK:
        raise ValueError("Esperado nums com 6 dezenas por linha.")

    if recent_draws and recent_draws > 0:
        # precisamos de pelo menos 2 linhas pra ter delta
        arr = arr[-(recent_draws + 1) :]

    if len(arr) < 2:
        # sem histórico suficiente
        empty = [dict() for _ in range(PICK)]
        return {"probs_by_pos": empty, "support_by_pos": [[] for _ in range(PICK)], "meta": {"n_pairs": 0, "alpha": alpha}}

    deltas = arr[1:, :] - arr[:-1, :]  # (n-1,6)

    probs_by_pos: List[Dict[int, float]] = []
    support_by_pos: List[List[int]] = []

    for j in range(PICK):
        c = Counter(int(x) for x in deltas[:, j].tolist())
        support = sorted(c.keys())
        support_by_pos.append(support)

        # suavização (Dirichlet/Laplace): (count+alpha)/(N+alpha*K)
        N = float(sum(c.values()))
        K = float(len(support)) if len(support) else 1.0

        probs = {}
        for d in support:
            probs[d] = (float(c[d]) + alpha) / (N + alpha * K)

        # fallback: se algo não observado, damos uma prob mínima
        probs_by_pos.append(probs)

    return {
        "probs_by_pos": probs_by_pos,
        "support_by_pos": support_by_pos,
        "meta": {
            "n_pairs": int(len(arr) - 1),
            "alpha": float(alpha),
            "recent_draws": int(recent_draws),
        },
    }


def transition_score(
    cand_nums: Sequence[int],
    last_nums: Sequence[int],
    model: Dict[str, object],
    *,
    floor: float = 1e-9,
) -> float:
    """
    Score = média do log P(delta_pos | pos), onde:
      delta_pos = cand_pos - last_pos
    Quanto maior (menos negativo), mais “compatível” com deltas históricos.
    """
    probs_by_pos: List[Dict[int, float]] = model.get("probs_by_pos", [])  # type: ignore
    if not probs_by_pos or len(probs_by_pos) != PICK:
        return float("nan")

    c = sorted(int(x) for x in cand_nums)
    l = sorted(int(x) for x in last_nums)

    logs = []
    for j in range(PICK):
        delta = int(c[j] - l[j])
        p = probs_by_pos[j].get(delta, None)
        if p is None:
            # delta nunca apareceu nessa posição → penaliza forte, mas não -inf
            p = floor
        logs.append(math.log(max(float(p), floor)))

    return float(sum(logs) / len(logs))