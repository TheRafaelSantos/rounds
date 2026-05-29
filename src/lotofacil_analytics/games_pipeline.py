from __future__ import annotations

import logging

import pandas as pd

from .config import AppConfig
from .games import GeneratedGamesSummary, generate_games
from .storage import load_processed_csv, sanitize_dataframe_for_tabular_output


class GeneratedGamesPipeline:
    def __init__(self, *, config: AppConfig, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    def run(
        self,
        *,
        method: str,
        qty: int,
        seed: int,
        window: int,
        candidates: int,
        candidate_pool: int,
        generations: int,
        population: int,
        draw_hour: int = 20,
        draw_minute: int = 0,
    ) -> GeneratedGamesSummary:
        concursos = load_processed_csv(self.config.processed_csv_path)
        if concursos.empty:
            raise ValueError("Historico local nao encontrado. Rode primeiro: python main.py --update")

        games = generate_games(
            concursos,
            method=method,
            qty=qty,
            seed=seed,
            window=window,
            candidates=candidates,
            candidate_pool=candidate_pool,
            generations=generations,
            population=population,
            draw_hour=draw_hour,
            draw_minute=draw_minute,
        )
        games = sanitize_dataframe_for_tabular_output(games)

        self.config.generated_games_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.generated_games_excel_path.parent.mkdir(parents=True, exist_ok=True)
        games.to_csv(self.config.generated_games_csv_path, index=False, encoding="utf-8-sig")

        resumo = pd.DataFrame(
            [
                {"campo": "metodo", "valor": method},
                {"campo": "qty", "valor": int(qty)},
                {"campo": "seed", "valor": int(seed)},
                {"campo": "draw_hour_brasilia", "valor": int(draw_hour)},
                {"campo": "draw_minute_brasilia", "valor": int(draw_minute)},
                {"campo": "ultimo_concurso_base", "valor": int(concursos["concurso"].max())},
            ]
        )
        with pd.ExcelWriter(self.config.generated_games_excel_path, engine="openpyxl") as writer:
            games.to_excel(writer, index=False, sheet_name="jogos_gerados")
            resumo.to_excel(writer, index=False, sheet_name="parametros")

        self.logger.info("Jogos gerados salvos em %s", self.config.generated_games_csv_path)
        self.logger.info("Excel de jogos gerados salvo em %s", self.config.generated_games_excel_path)
        return GeneratedGamesSummary(
            rows=int(len(games)),
            method=method,
            csv_path=str(self.config.generated_games_csv_path),
            excel_path=str(self.config.generated_games_excel_path),
        )
