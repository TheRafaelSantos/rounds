from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .backtest_lotofacil import compute_hits
from .dezenas_history import build_dezenas_historico


FEATURE_COLUMNS = [
    "concursos_anteriores",
    "freq_total_ate_anterior",
    "freq_ultimos_5",
    "freq_ultimos_10",
    "freq_ultimos_20",
    "freq_ultimos_50",
    "freq_ultimos_100",
    "saiu_concurso_anterior",
    "saiu_ultimos_5",
    "saiu_ultimos_10",
    "atraso_atual",
    "nunca_saiu_ate_anterior",
    "media_atraso_ate_anterior",
    "rank_freq_total",
    "rank_freq_50",
    "rank_atraso",
    "dezena_par",
    "dezena_prima",
    "dezena_fibonacci",
    "dezena_quadrado_perfeito",
    "grupo_dezena_5",
    "linha_volante",
    "coluna_volante",
]


@dataclass(frozen=True)
class MLSummary:
    dataset_rows: int
    predictions_rows: int
    summary_rows: int
    first_test_concurso: int
    last_test_concurso: int
    dataset_csv_path: str
    predictions_csv_path: str
    summary_csv_path: str
    excel_path: str

    def to_console(self) -> str:
        return "\n".join(
            [
                "",
                "Resumo Lotofacil Analytics - Fase 7",
                "Acao: ml",
                f"Linhas dataset: {self.dataset_rows}",
                f"Linhas predicoes: {self.predictions_rows}",
                f"Linhas resumo: {self.summary_rows}",
                f"Primeiro concurso de teste: {self.first_test_concurso}",
                f"Ultimo concurso de teste: {self.last_test_concurso}",
                f"CSV dataset: {self.dataset_csv_path}",
                f"CSV predicoes: {self.predictions_csv_path}",
                f"CSV resumo: {self.summary_csv_path}",
                f"Excel: {self.excel_path}",
                "Mensagem: ML temporal executado com split por tempo e comparado contra baselines.",
            ]
        )


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _prepare_matrix(df: pd.DataFrame, means: pd.Series | None = None, stds: pd.Series | None = None) -> Tuple[np.ndarray, pd.Series, pd.Series]:
    xdf = df[FEATURE_COLUMNS].copy()
    xdf = xdf.fillna(0.0).astype(float)
    if means is None:
        means = xdf.mean()
    if stds is None:
        stds = xdf.std(ddof=0).replace(0, 1.0)
    x = ((xdf - means) / stds).to_numpy(dtype=float)
    intercept = np.ones((x.shape[0], 1), dtype=float)
    return np.hstack([intercept, x]), means, stds


def _train_logistic(train_df: pd.DataFrame, *, epochs: int, learning_rate: float, l2: float) -> Tuple[np.ndarray, pd.Series, pd.Series]:
    x, means, stds = _prepare_matrix(train_df)
    y = train_df["saiu_no_concurso"].to_numpy(dtype=float)
    weights = np.zeros(x.shape[1], dtype=float)
    n = max(1, x.shape[0])
    for _ in range(max(1, int(epochs))):
        pred = _sigmoid(x @ weights)
        grad = (x.T @ (pred - y)) / n
        grad[1:] += float(l2) * weights[1:]
        weights -= float(learning_rate) * grad
    return weights, means, stds


def _predict_logistic(df: pd.DataFrame, weights: np.ndarray, means: pd.Series, stds: pd.Series) -> np.ndarray:
    x, _, _ = _prepare_matrix(df, means, stds)
    return _sigmoid(x @ weights)


def _split_contests(dataset: pd.DataFrame, *, train_ratio: float, validation_ratio: float) -> pd.DataFrame:
    contests = sorted(int(c) for c in dataset["concurso"].unique())
    if len(contests) < 5:
        raise ValueError("Dataset pequeno demais para split temporal de ML.")
    train_end = max(1, int(len(contests) * float(train_ratio)))
    valid_end = max(train_end + 1, int(len(contests) * (float(train_ratio) + float(validation_ratio))))
    valid_end = min(valid_end, len(contests) - 1)
    train_set = set(contests[:train_end])
    valid_set = set(contests[train_end:valid_end])
    test_set = set(contests[valid_end:])
    out = dataset.copy()
    out["split"] = out["concurso"].map(lambda c: "train" if int(c) in train_set else ("validation" if int(c) in valid_set else "test"))
    if not test_set:
        raise ValueError("Split temporal ficou sem concursos de teste.")
    return out


def build_ml_dataset(concursos: pd.DataFrame) -> pd.DataFrame:
    historico = build_dezenas_historico(concursos)
    missing = [col for col in FEATURE_COLUMNS + ["saiu_no_concurso"] if col not in historico.columns]
    if missing:
        raise ValueError(f"Dataset ML sem colunas obrigatorias: {missing}")
    return historico


def _format_nums(nums: Sequence[int]) -> str:
    return " ".join(f"{int(n):02d}" for n in sorted(nums))


def _pick_top15(group: pd.DataFrame, score_col: str, *, ascending: bool = False) -> List[int]:
    ranked = group.sort_values([score_col, "dezena"], ascending=[ascending, True])
    return sorted(int(n) for n in ranked.head(15)["dezena"].tolist())


def _random_pick(group: pd.DataFrame, *, seed: int) -> List[int]:
    concurso = int(group["concurso"].iloc[0])
    rng = random.Random(int(seed) * 1_000_003 + concurso)
    return sorted(rng.sample([int(n) for n in group["dezena"].tolist()], 15))


def _evaluate_predictions(dataset: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    eval_df = dataset[dataset["split"].isin(["validation", "test"])].copy()
    for (split, concurso), group in eval_df.groupby(["split", "concurso"], sort=True):
        real = sorted(int(row["dezena"]) for _, row in group[group["saiu_no_concurso"] == 1].iterrows())
        picks = {
            "ml_logistico_simples": _pick_top15(group, "prob_ml", ascending=False),
            "baseline_freq_100": _pick_top15(group, "freq_ultimos_100", ascending=False),
            "baseline_atraso": _pick_top15(group, "atraso_atual", ascending=False),
            "baseline_random": _random_pick(group, seed=seed),
        }
        for model, nums in picks.items():
            hits = compute_hits(nums, real)
            rows.append(
                {
                    "split": split,
                    "modelo_nome": model,
                    "concurso_previsto": int(concurso),
                    "data_sorteio": str(group["data_sorteio"].iloc[0]),
                    "numeros_sugeridos": _format_nums(nums),
                    "numeros_reais": _format_nums(real),
                    "qtd_acertos": int(hits),
                    "acertou_11": int(hits >= 11),
                    "acertou_12": int(hits >= 12),
                    "acertou_13": int(hits >= 13),
                    "acertou_14": int(hits >= 14),
                    "acertou_15": int(hits >= 15),
                }
            )
    return pd.DataFrame(rows)


def _summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (split, model), group in predictions.groupby(["split", "modelo_nome"]):
        hits = group["qtd_acertos"].astype(int)
        rows.append(
            {
                "split": split,
                "modelo_nome": model,
                "n_jogos": int(len(group)),
                "media_acertos": round(float(hits.mean()), 6),
                "min_acertos": int(hits.min()),
                "max_acertos": int(hits.max()),
                "p_acertou_11": round(float((hits >= 11).mean()), 6),
                "p_acertou_12": round(float((hits >= 12).mean()), 6),
                "p_acertou_13": round(float((hits >= 13).mean()), 6),
                "p_acertou_14": round(float((hits >= 14).mean()), 6),
                "p_acertou_15": round(float((hits >= 15).mean()), 6),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "media_acertos", "modelo_nome"], ascending=[True, False, True]).reset_index(drop=True)


def run_ml_temporal(
    concursos: pd.DataFrame,
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    epochs: int = 400,
    learning_rate: float = 0.05,
    l2: float = 0.001,
    seed: int = 123,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset = build_ml_dataset(concursos)
    dataset = _split_contests(dataset, train_ratio=train_ratio, validation_ratio=validation_ratio)
    train_df = dataset[dataset["split"] == "train"].copy()
    weights, means, stds = _train_logistic(train_df, epochs=epochs, learning_rate=learning_rate, l2=l2)
    dataset["prob_ml"] = _predict_logistic(dataset, weights, means, stds)
    predictions = _evaluate_predictions(dataset, seed=seed)
    summary = _summarize_predictions(predictions)
    return dataset, predictions, summary
