from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lotofacil_analytics.config import AppConfig
from lotofacil_analytics.auditoria_pipeline import AuditoriaPipeline
from lotofacil_analytics.backtest_pipeline import BacktestPipeline
from lotofacil_analytics.combinacoes_pipeline import CombinacoesPipeline
from lotofacil_analytics.dezenas_pipeline import DezenasPipeline
from lotofacil_analytics.features_pipeline import FeaturePipeline
from lotofacil_analytics.logger import setup_logger
from lotofacil_analytics.ml_pipeline import MLPipeline
from lotofacil_analytics.optimizer_pipeline import OptimizerPipeline
from lotofacil_analytics.pipeline import LotofacilPipeline
from lotofacil_analytics.predictor_pipeline import PredictorPipeline


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
    mode.add_argument("--audit", action="store_true", help="Executa auditoria estatistica exploratoria da Fase 6.")
    mode.add_argument("--ml", action="store_true", help="Executa ML temporal leve da Fase 7.")
    mode.add_argument("--optimize", action="store_true", help="Gera candidatos otimizados da Fase 8.")
    mode.add_argument("--predict", action="store_true", help="Gera exatamente 2 jogos finais da Fase 9.")

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
    parser.add_argument("--monte-carlo-runs", type=int, default=500, help="Rodadas de Monte Carlo para auditoria.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Proporcao inicial dos concursos usada para treino ML.")
    parser.add_argument("--validation-ratio", type=float, default=0.15, help="Proporcao apos treino usada para validacao ML.")
    parser.add_argument("--epochs", type=int, default=400, help="Epocas da regressao logistica simples.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Taxa de aprendizado da regressao logistica simples.")
    parser.add_argument("--l2", type=float, default=0.001, help="Regularizacao L2 da regressao logistica simples.")
    parser.add_argument("--candidate-pool", type=int, default=10000, help="Quantidade de candidatos Monte Carlo para otimizacao.")
    parser.add_argument("--top-games", type=int, default=100, help="Quantidade de candidatos finais a salvar.")
    parser.add_argument("--generations", type=int, default=20, help="Geracoes do genetico simples.")
    parser.add_argument("--population", type=int, default=80, help="Populacao por geracao do genetico simples.")
    parser.add_argument("--max-overlap-final", type=int, default=10, help="Overlap maximo desejado entre os 2 jogos finais.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not (
        args.update
        or args.full
        or args.status
        or args.features
        or args.dezenas
        or args.combinacoes
        or args.backtest
        or args.audit
        or args.ml
        or args.optimize
        or args.predict
    ):
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
        if args.predict:
            summary = PredictorPipeline(config=config, logger=logger).predict(
                seed=args.seed,
                candidate_pool=args.candidate_pool,
                top_games=args.top_games,
                generations=args.generations,
                population=args.population,
                max_overlap=args.max_overlap_final,
            )
        elif args.optimize:
            summary = OptimizerPipeline(config=config, logger=logger).run(
                seed=args.seed,
                candidate_pool=args.candidate_pool,
                top_games=args.top_games,
                generations=args.generations,
                population=args.population,
            )
        elif args.ml:
            summary = MLPipeline(config=config, logger=logger).run(
                train_ratio=args.train_ratio,
                validation_ratio=args.validation_ratio,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                l2=args.l2,
                seed=args.seed,
            )
        elif args.audit:
            summary = AuditoriaPipeline(config=config, logger=logger).run(
                monte_carlo_runs=args.monte_carlo_runs,
                seed=args.seed,
            )
        elif args.backtest:
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
