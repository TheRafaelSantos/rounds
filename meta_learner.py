# meta_learner.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable


@dataclass
class HedgeMetaLearner:
    """
    Meta-aprendizado por Exponential Weights (Hedge).
    - Mantém pesos por bucket/estratégia.
    - Atualiza pesos com base em um reward (ex.: hits).
    - Pode ter "forget" (esquecimento) para priorizar o recente.
    """
    weights: Dict[str, float]
    eta: float = 0.25       # taxa de aprendizado
    forget: float = 0.0     # 0..1 (0 = sem esquecimento)
    min_w: float = 1e-12

    @classmethod
    def init(cls, buckets: Iterable[str], *, eta: float = 0.25, forget: float = 0.0) -> "HedgeMetaLearner":
        w = {str(b): 1.0 for b in buckets}
        obj = cls(weights=w, eta=float(eta), forget=float(forget))
        obj.normalize()
        return obj

    def normalize(self) -> None:
        s = sum(max(self.min_w, float(v)) for v in self.weights.values())
        if s <= 0:
            n = max(1, len(self.weights))
            self.weights = {k: 1.0 / n for k in self.weights}
            return
        self.weights = {k: max(self.min_w, float(v)) / s for k, v in self.weights.items()}

    def update(self, rewards: Dict[str, float]) -> None:
        # Esquecimento leve: puxa pesos para uniforme (melhora robustez)
        if self.forget and self.forget > 0 and len(self.weights) > 0:
            n = len(self.weights)
            uni = 1.0 / n
            for k in list(self.weights.keys()):
                self.weights[k] = (1.0 - self.forget) * float(self.weights[k]) + self.forget * uni

        # Update exponencial
        for k, r in rewards.items():
            k = str(k)
            if k not in self.weights:
                self.weights[k] = 1.0
            self.weights[k] *= math.exp(self.eta * float(r))

        self.normalize()

    def meta_score(self, bucket: str, base_score: float) -> float:
        return float(self.weights.get(str(bucket), 0.0)) * float(base_score)
