from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lotofacil_analytics.config import AppConfig
from lotofacil_analytics.backtest_pipeline import BacktestPipeline
from lotofacil_analytics.combinacoes_pipeline import CombinacoesPipeline
from lotofacil_analytics.dezenas_pipeline import DezenasPipeline
from lotofacil_analytics.features_pipeline import FeaturePipeline
from lotofacil_analytics.logger import setup_logger
from lotofacil_analytics.pipeline import LotofacilPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lotofacil Analytics - historico, features e proximas fases."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--update", action="store_true", help="Atualiza incrementalmente a base local.")
    mode.add_argument("--full", action="store_true", help="Rebaixa todo o historico desde o concurso 1.")
    mode.add_argument("--status", action="store_true", help="Mostra o estado local sem consultar todos os concursos.")
    mode.add_argument("--features", action="store_true", help="Gera features basicas da Fase 2 a partir da base local.")
    mode.add_argument("--dezenas", action="store_true", help="Gera historico por dezena da Fase 3.")
    mode.add_argument("--combinacoes", action="store_true", help="Gera combinacoes e assinaturas da Fase 4.")
    mode.add_argument("--backtest", action="store_true", help="Executa backtest walk-forward da Fase 5.")

    parser.add_argument("--base-dir", default=".", help="Pasta raiz do projeto.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Timeout por requisicao HTTP, em segundos.")
    parser.add_argument("--retries", type=int, default=3, help="Tentativas por concurso em caso de erro temporario.")
    parser.add_argument("--sleep", type=float, default=0.05, help="Pausa entre requisicoes, em segundos.")
    parser.add_argument("--from-concurso", type=int, default=None, help="Inicio manual do intervalo a baixar.")
    parser.add_argument("--to-concurso", type=int, default=None, help="Fim manual do intervalo a baixar.")
    parser.add_argument("--n-eval", type=int, default=300, help="Quantidade de concursos finais para avaliar no backtest.")
    parser.add_argument("--min-history", type=int, default=300, help="Historico minimo antes de iniciar o backtest.")
    parser.add_argument("--seed", type=int, default=123, help="Seed para metodos randomicos.")
    parser.add_argument("--window", type=int, default=100, help="Janela de frequencia recente para metodos simples.")
    parser.add_argument("--candidates", type=int, default=1000, help="Candidatos aleatorios avaliados pelo balanceado_basico.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not (args.update or args.full or args.status or args.features or args.dezenas or args.combinacoes or args.backtest):
        parser.print_help()
        return 2

    config = AppConfig.from_base_dir(
        Path(args.base_dir).resolve(),
        timeout_seconds=args.timeout,
        max_retries=args.retries,
        request_sleep_seconds=args.sleep,
    )
    logger = setup_logger(config.logs_dir)
    pipeline = LotofacilPipeline(config=config, logger=logger)

    try:
        if args.backtest:
            summary = BacktestPipeline(config=config, logger=logger).run(
                n_eval=args.n_eval,
                min_history=args.min_history,
                seed=args.seed,
                window=args.window,
                candidates=args.candidates,
            )
        elif args.combinacoes:
            summary = CombinacoesPipeline(config=config, logger=logger).build_combinacoes()
        elif args.dezenas:
            summary = DezenasPipeline(config=config, logger=logger).build_history()
        elif args.features:
            summary = FeaturePipeline(config=config, logger=logger).build_base_features()
        elif args.status:
            summary = pipeline.status()
        else:
            summary = pipeline.update(
                force_full=bool(args.full),
                from_concurso=args.from_concurso,
                to_concurso=args.to_concurso,
            )
    except Exception as exc:
        logger.exception("Falha na execucao: %s", exc)
        print(f"ERRO: {exc}")
        return 1

    print(summary.to_console())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
