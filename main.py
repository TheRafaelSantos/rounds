from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lotofacil_analytics.config import AppConfig
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

    parser.add_argument("--base-dir", default=".", help="Pasta raiz do projeto.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Timeout por requisicao HTTP, em segundos.")
    parser.add_argument("--retries", type=int, default=3, help="Tentativas por concurso em caso de erro temporario.")
    parser.add_argument("--sleep", type=float, default=0.05, help="Pausa entre requisicoes, em segundos.")
    parser.add_argument("--from-concurso", type=int, default=None, help="Inicio manual do intervalo a baixar.")
    parser.add_argument("--to-concurso", type=int, default=None, help="Fim manual do intervalo a baixar.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not (args.update or args.full or args.status or args.features):
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
        if args.features:
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
