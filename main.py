from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lotofacil_analytics.config import AppConfig
from lotofacil_analytics.auditoria_pipeline import AuditoriaPipeline
from lotofacil_analytics.build_executable import build_executable
from lotofacil_analytics.backtest_pipeline import BacktestPipeline
from lotofacil_analytics.calibration_lab_pipeline import CalibrationLabPipeline
from lotofacil_analytics.calibration_pilot_pipeline import CalibrationPilotPipeline
from lotofacil_analytics.climate_pipeline import ClimatePipeline
from lotofacil_analytics.combinacoes_pipeline import CombinacoesPipeline
from lotofacil_analytics.dezenas_pipeline import DezenasPipeline
from lotofacil_analytics.decision_layer import WEIGHT_PROFILE_PRESETS
from lotofacil_analytics.decision_layer_pipeline import DecisionLayerPipeline
from lotofacil_analytics.engine_calibration_pipeline import EngineCalibrationPipeline
from lotofacil_analytics.export_full import export_full_workbook
from lotofacil_analytics.final_backtest_pipeline import FinalBacktestPipeline
from lotofacil_analytics.features_pipeline import FeaturePipeline
from lotofacil_analytics.games_pipeline import GeneratedGamesPipeline
from lotofacil_analytics.logger import setup_logger
from lotofacil_analytics.mandel_pipeline import MandelPipeline
from lotofacil_analytics.ml_pipeline import MLPipeline
from lotofacil_analytics.optimizer_pipeline import OptimizerPipeline
from lotofacil_analytics.pipeline import LotofacilPipeline
from lotofacil_analytics.post_result_analysis import analyze_post_result
from lotofacil_analytics.predictor_pipeline import PredictorPipeline
from lotofacil_analytics.interface_web import run_web_server
from lotofacil_analytics.supervised_calibration_pipeline import SupervisedCalibrationPipeline
from lotofacil_analytics.transition_pipeline import TransitionPipeline
from lotofacil_analytics.temporal_deep_pipeline import TemporalDeepPipeline


def _safe_file_label(label: str) -> str:
    safe = "".join(
        ch if ch.isascii() and (ch.isalnum() or ch in {"-", "_"}) else "_"
        for ch in str(label).strip()
    ).strip("_")
    return safe or "resultado"


def _path_with_label(path: Path, label: str) -> Path:
    safe = _safe_file_label(label)
    return path.with_name(f"{path.stem}_{safe}{path.suffix}")


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
    mode.add_argument("--transitions", action="store_true", help="Analisa transicoes concurso N contra N+1.")
    mode.add_argument("--climate", action="store_true", help="Baixa/gera features climaticas historicas por localidade e horario.")
    mode.add_argument("--temporal-deep", action="store_true", help="Gera analises temporais profundas incrementais.")
    mode.add_argument("--backtest", action="store_true", help="Executa backtest walk-forward da Fase 5.")
    mode.add_argument("--audit", action="store_true", help="Executa auditoria estatistica exploratoria da Fase 6.")
    mode.add_argument("--ml", action="store_true", help="Executa ML temporal leve da Fase 7.")
    mode.add_argument("--optimize", action="store_true", help="Gera candidatos otimizados da Fase 8.")
    mode.add_argument("--predict", action="store_true", help="Gera exatamente 2 jogos finais da Fase 9.")
    mode.add_argument("--predict-single", action="store_true", help="Gera 1 jogo unico pela camada superior de decisao.")
    mode.add_argument("--mandel", action="store_true", help="Gera plano Mandel/bolao com desdobramento e fechamento reduzido.")
    mode.add_argument("--backtest-exhaustive", action="store_true", help="Backtest walk-forward do jogo unico exaustivo.")
    mode.add_argument("--ablation-test", action="store_true", help="Mede impacto de remover familias de score do motor exaustivo.")
    mode.add_argument("--tune-weights", action="store_true", help="Testa perfis de pesos e salva o melhor perfil observado.")
    mode.add_argument("--calibrate-engine", action="store_true", help="Calibra pesos do motor por walk-forward contra concursos passados.")
    mode.add_argument("--calibration-lab", action="store_true", help="Roda calibracao 24/7 retomavel buscando 15 pontos em um dos dois jogos.")
    mode.add_argument("--supervised-calibration", action="store_true", help="Roda aprendizado supervisionado com gabarito historico e aplica pesos medios ao motor.")
    mode.add_argument("--calibration-pilot", action="store_true", help="Roda piloto retomavel de busca de pesos por resultado final.")
    mode.add_argument("--analyze-result", action="store_true", help="Analisa resultado real contra os jogos previstos.")
    mode.add_argument("--final-backtest", action="store_true", help="Executa backtest do score final completo contra aleatorio.")
    mode.add_argument("--export", action="store_true", help="Gera Excel consolidado com as abas do briefing.")
    mode.add_argument("--generate-games", action="store_true", help="Gera jogos por metodo especifico para estudo/backtesting manual.")
    mode.add_argument("--serve", action="store_true", help="Inicia interface web local da Fase 10.")
    mode.add_argument("--build-exe", action="store_true", help="Gera executavel Windows com PyInstaller, se instalado.")

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
    parser.add_argument("--max-overlap-final", type=int, default=8, help="Overlap maximo desejado entre os 2 jogos finais.")
    parser.add_argument("--mode", choices=["rapido", "completo", "experimental"], default="rapido", help="Modo do --predict.")
    parser.add_argument("--engine", choices=["exaustivo", "heuristico"], default="exaustivo", help="Motor de score para --optimize e --predict.")
    parser.add_argument("--exhaustive-limit", type=int, default=None, help="Limite tecnico para testar o motor exaustivo; omitido avalia todas as combinacoes.")
    parser.add_argument("--weight-profile", choices=sorted(WEIGHT_PROFILE_PRESETS), default="padrao_atual", help="Perfil de pesos da camada superior.")
    parser.add_argument("--mandel-universe-size", type=int, default=18, help="Quantidade de dezenas no universo Mandel, entre 15 e 20.")
    parser.add_argument("--mandel-guarantee-hits", type=int, default=14, help="Garantia condicional alvo do fechamento reduzido, entre 11 e 15.")
    parser.add_argument("--mandel-max-reduced-games", type=int, default=80, help="Maximo de jogos no fechamento reduzido Mandel.")
    parser.add_argument("--validation-n-eval", type=int, default=3, help="Concursos finais avaliados nos comandos de validacao da camada superior.")
    parser.add_argument("--method", default="balanceado_basico", help="Metodo do --generate-games.")
    parser.add_argument("--qty", type=int, default=10, help="Quantidade de jogos do --generate-games.")
    parser.add_argument("--draw-hour", type=int, default=20, help="Hora de Brasilia usada para contexto lunar do proximo sorteio.")
    parser.add_argument("--draw-minute", type=int, default=0, help="Minuto de Brasilia usado para contexto lunar do proximo sorteio.")
    parser.add_argument("--climate-max-locations", type=int, default=0, help="Limite de cidades/UF no --climate; 0 processa todas.")
    parser.add_argument("--climate-force", action="store_true", help="Ignora cache local do Open-Meteo no --climate.")
    parser.add_argument("--calibration-from-concurso", type=int, default=2500, help="Concurso inicial do --calibrate-engine.")
    parser.add_argument("--calibration-to-concurso", type=int, default=None, help="Concurso final do --calibrate-engine; omitido usa ultimo local.")
    parser.add_argument("--calibration-baseline-samples", type=int, default=30, help="Combos aleatorios por concurso na calibracao.")
    parser.add_argument("--lab-from-concurso", type=int, default=2500, help="Concurso inicial do --calibration-lab.")
    parser.add_argument("--lab-to-concurso", type=int, default=None, help="Concurso final do --calibration-lab; omitido usa ultimo local.")
    parser.add_argument("--lab-max-attempts", type=int, default=0, help="Maximo de tentativas nesta execucao do --calibration-lab; 0 roda sem limite.")
    parser.add_argument("--lab-top-games", type=int, default=5000, help="Quantidade de candidatos preservados por tentativa do --calibration-lab.")
    parser.add_argument("--lab-exhaustive-limit", type=int, default=0, help="Limite tecnico de combinacoes por tentativa; 0 avalia todas.")
    parser.add_argument("--lab-max-runtime-seconds", type=int, default=0, help="Tempo maximo desta execucao do --calibration-lab; 0 sem limite.")
    parser.add_argument("--lab-reset", action="store_true", help="Apaga a memoria anterior do --calibration-lab.")
    parser.add_argument("--supervised-from-concurso", type=int, default=2500, help="Concurso inicial do --supervised-calibration.")
    parser.add_argument("--supervised-to-concurso", type=int, default=None, help="Concurso final do --supervised-calibration; omitido usa ultimo local.")
    parser.add_argument("--supervised-samples", type=int, default=800, help="Amostras de combinacoes por concurso no aprendizado supervisionado.")
    parser.add_argument("--supervised-max-contests", type=int, default=0, help="Maximo de concursos nesta execucao do --supervised-calibration; 0 processa todos.")
    parser.add_argument("--supervised-reset", action="store_true", help="Apaga a memoria anterior do --supervised-calibration.")
    parser.add_argument("--supervised-loop", action="store_true", help="Mantem o --supervised-calibration rodando em ciclos retomaveis.")
    parser.add_argument("--supervised-sleep-seconds", type=int, default=30, help="Pausa entre ciclos do --supervised-loop.")
    parser.add_argument("--pilot-concurso", type=int, default=2500, help="Concurso-alvo do --calibration-pilot.")
    parser.add_argument("--pilot-games", type=int, default=100, help="Quantidade de tentativas de pesos/jogos no --calibration-pilot.")
    parser.add_argument("--pilot-candidate-pool", type=int, default=5000, help="Quantidade de candidatos base reaproveitados no piloto.")
    parser.add_argument("--pilot-exhaustive-limit", type=int, default=0, help="Limite de combinacoes na base do piloto; 0 avalia todas.")
    parser.add_argument("--pilot-reset", action="store_true", help="Apaga memoria anterior do concurso no --calibration-pilot.")
    parser.add_argument("--actual-numbers", default=None, help="15 dezenas sorteadas para --analyze-result.")
    parser.add_argument("--result-label", default="resultado", help="Rotulo do resultado analisado.")
    parser.add_argument("--result-concurso", type=int, default=None, help="Numero do concurso analisado em --analyze-result.")
    parser.add_argument("--prediction-file", default=None, help="CSV de previsao a comparar; se omitido, usa a previsao padrao.")
    parser.add_argument("--final-n-eval", type=int, default=60, help="Concursos finais avaliados em --final-backtest.")
    parser.add_argument("--final-candidate-pool", type=int, default=2500, help="Candidatos por concurso no --final-backtest.")
    parser.add_argument("--final-generations", type=int, default=6, help="Geracoes por concurso no --final-backtest.")
    parser.add_argument("--final-population", type=int, default=40, help="Populacao por concurso no --final-backtest.")
    parser.add_argument("--host", default="127.0.0.1", help="Host da interface web local.")
    parser.add_argument("--port", type=int, default=8765, help="Porta da interface web local.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not (0 <= int(args.draw_hour) <= 23 and 0 <= int(args.draw_minute) <= 59):
        parser.error("--draw-hour deve estar entre 0 e 23 e --draw-minute entre 0 e 59.")
    if not (15 <= int(args.mandel_universe_size) <= 20):
        parser.error("--mandel-universe-size deve estar entre 15 e 20.")
    if not (11 <= int(args.mandel_guarantee_hits) <= 15):
        parser.error("--mandel-guarantee-hits deve estar entre 11 e 15.")
    if int(args.mandel_max_reduced_games) <= 0:
        parser.error("--mandel-max-reduced-games deve ser maior que zero.")

    if not (
        args.update
        or args.full
        or args.status
        or args.features
        or args.dezenas
        or args.combinacoes
        or args.transitions
        or args.climate
        or args.temporal_deep
        or args.backtest
        or args.audit
        or args.ml
        or args.optimize
        or args.predict
        or args.predict_single
        or args.mandel
        or args.backtest_exhaustive
        or args.ablation_test
        or args.tune_weights
        or args.calibrate_engine
        or args.calibration_lab
        or args.supervised_calibration
        or args.calibration_pilot
        or args.analyze_result
        or args.final_backtest
        or args.export
        or args.generate_games
        or args.serve
        or args.build_exe
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
        if args.serve:
            run_web_server(config=config, logger=logger, host=args.host, port=args.port)
            return 0
        if args.build_exe:
            summary = build_executable(config.base_dir)
        elif args.analyze_result:
            if not args.actual_numbers:
                raise ValueError("Informe --actual-numbers com as 15 dezenas sorteadas.")
            prediction_path = Path(args.prediction_file).resolve() if args.prediction_file else config.prediction_csv_path
            result_label = _safe_file_label(args.result_label)
            summary = analyze_post_result(
                actual_numbers=args.actual_numbers,
                predictions_path=prediction_path,
                optimizer_candidates_path=config.optimizer_candidates_csv_path,
                games_csv_path=_path_with_label(config.post_result_games_csv_path, result_label),
                dezenas_csv_path=_path_with_label(config.post_result_dezenas_csv_path, result_label),
                report_path=_path_with_label(config.post_result_report_path, result_label),
                excel_path=_path_with_label(config.post_result_excel_path, result_label),
                label=args.result_label,
                concurso=args.result_concurso,
            )
        elif args.final_backtest:
            summary = FinalBacktestPipeline(config=config, logger=logger).run(
                n_eval=args.final_n_eval,
                min_history=args.min_history,
                seed=args.seed,
                candidate_pool=args.final_candidate_pool,
                top_games=args.top_games,
                generations=args.final_generations,
                population=args.final_population,
                max_overlap=args.max_overlap_final,
                draw_hour=args.draw_hour,
                draw_minute=args.draw_minute,
            )
        elif args.export:
            summary = export_full_workbook(config=config, logger=logger)
        elif args.generate_games:
            summary = GeneratedGamesPipeline(config=config, logger=logger).run(
                method=args.method,
                qty=args.qty,
                seed=args.seed,
                window=args.window,
                candidates=args.candidates,
                candidate_pool=args.candidate_pool,
                generations=args.generations,
                population=args.population,
                draw_hour=args.draw_hour,
                draw_minute=args.draw_minute,
            )
        elif args.climate:
            summary = ClimatePipeline(config=config, logger=logger).run(
                draw_hour=args.draw_hour,
                draw_minute=args.draw_minute,
                max_locations=args.climate_max_locations,
                force=args.climate_force,
            )
        elif args.temporal_deep:
            summary = TemporalDeepPipeline(config=config, logger=logger).run(force=False)
        elif args.predict_single:
            summary = DecisionLayerPipeline(config=config, logger=logger).predict_single(
                seed=args.seed,
                candidate_pool=args.candidate_pool,
                top_games=args.top_games,
                generations=args.generations,
                population=args.population,
                draw_hour=args.draw_hour,
                draw_minute=args.draw_minute,
                engine=args.engine,
                exhaustive_limit=args.exhaustive_limit,
                weight_profile=args.weight_profile,
            )
        elif args.mandel:
            summary = MandelPipeline(config=config, logger=logger).run(
                universe_size=args.mandel_universe_size,
                guarantee_hits=args.mandel_guarantee_hits,
                max_reduced_games=args.mandel_max_reduced_games,
                draw_hour=args.draw_hour,
                draw_minute=args.draw_minute,
            )
        elif args.backtest_exhaustive:
            summary = DecisionLayerPipeline(config=config, logger=logger).backtest_exhaustive(
                n_eval=args.validation_n_eval,
                min_history=args.min_history,
                top_games=args.top_games,
                draw_hour=args.draw_hour,
                draw_minute=args.draw_minute,
                exhaustive_limit=args.exhaustive_limit,
                weight_profile=args.weight_profile,
            )
        elif args.ablation_test:
            summary = DecisionLayerPipeline(config=config, logger=logger).ablation_test(
                n_eval=args.validation_n_eval,
                min_history=args.min_history,
                top_games=args.top_games,
                draw_hour=args.draw_hour,
                draw_minute=args.draw_minute,
                exhaustive_limit=args.exhaustive_limit,
            )
        elif args.tune_weights:
            summary = DecisionLayerPipeline(config=config, logger=logger).tune_weights(
                n_eval=args.validation_n_eval,
                min_history=args.min_history,
                top_games=args.top_games,
                draw_hour=args.draw_hour,
                draw_minute=args.draw_minute,
                exhaustive_limit=args.exhaustive_limit,
            )
        elif args.calibrate_engine:
            summary = EngineCalibrationPipeline(config=config, logger=logger).run(
                from_concurso=args.calibration_from_concurso,
                to_concurso=args.calibration_to_concurso,
                baseline_samples=args.calibration_baseline_samples,
                seed=args.seed,
                draw_hour=args.draw_hour,
                draw_minute=args.draw_minute,
            )
        elif args.calibration_lab:
            summary = CalibrationLabPipeline(config=config, logger=logger).run(
                from_concurso=args.lab_from_concurso,
                to_concurso=args.lab_to_concurso,
                max_attempts=args.lab_max_attempts,
                top_games=args.lab_top_games,
                exhaustive_limit=args.lab_exhaustive_limit if int(args.lab_exhaustive_limit) > 0 else None,
                max_overlap=args.max_overlap_final,
                seed=args.seed,
                draw_hour=args.draw_hour,
                draw_minute=args.draw_minute,
                min_history=args.min_history,
                max_runtime_seconds=args.lab_max_runtime_seconds,
                reset=args.lab_reset,
            )
        elif args.supervised_calibration:
            pipeline_supervised = SupervisedCalibrationPipeline(config=config, logger=logger)
            if args.supervised_loop:
                reset_next_cycle = bool(args.supervised_reset)
                while True:
                    summary = pipeline_supervised.run(
                        from_concurso=args.supervised_from_concurso,
                        to_concurso=args.supervised_to_concurso,
                        samples=args.supervised_samples,
                        max_contests=args.supervised_max_contests,
                        seed=args.seed,
                        draw_hour=args.draw_hour,
                        draw_minute=args.draw_minute,
                        min_history=args.min_history,
                        reset=reset_next_cycle,
                    )
                    print(summary.to_console(), flush=True)
                    reset_next_cycle = False
                    sleep_seconds = max(5, int(args.supervised_sleep_seconds))
                    if summary.status == "complete":
                        logger.info("Calibracao supervisionada completa; aguardando %s segundos.", sleep_seconds)
                    time.sleep(sleep_seconds)
            summary = pipeline_supervised.run(
                from_concurso=args.supervised_from_concurso,
                to_concurso=args.supervised_to_concurso,
                samples=args.supervised_samples,
                max_contests=args.supervised_max_contests,
                seed=args.seed,
                draw_hour=args.draw_hour,
                draw_minute=args.draw_minute,
                min_history=args.min_history,
                reset=args.supervised_reset,
            )
        elif args.calibration_pilot:
            summary = CalibrationPilotPipeline(config=config, logger=logger).run(
                target_concurso=args.pilot_concurso,
                attempts=args.pilot_games,
                candidate_pool=args.pilot_candidate_pool,
                exhaustive_limit=args.pilot_exhaustive_limit if int(args.pilot_exhaustive_limit) > 0 else None,
                max_overlap=args.max_overlap_final,
                seed=args.seed,
                draw_hour=args.draw_hour,
                draw_minute=args.draw_minute,
                reset=args.pilot_reset,
            )
        elif args.predict:
            if args.mode == "completo":
                pipeline.update(force_full=False, from_concurso=args.from_concurso, to_concurso=args.to_concurso)
                FeaturePipeline(config=config, logger=logger).build_base_features()
                DezenasPipeline(config=config, logger=logger).build_history()
                CombinacoesPipeline(config=config, logger=logger).build_combinacoes()
                ClimatePipeline(config=config, logger=logger).run(
                    draw_hour=args.draw_hour,
                    draw_minute=args.draw_minute,
                    max_locations=args.climate_max_locations,
                    force=args.climate_force,
                )
                TemporalDeepPipeline(config=config, logger=logger).run(force=False)
                BacktestPipeline(config=config, logger=logger).run(
                    n_eval=args.n_eval,
                    min_history=args.min_history,
                    seed=args.seed,
                    window=args.window,
                    candidates=args.candidates,
                )
                AuditoriaPipeline(config=config, logger=logger).run(
                    monte_carlo_runs=args.monte_carlo_runs,
                    seed=args.seed,
                )
                MLPipeline(config=config, logger=logger).run(
                    train_ratio=args.train_ratio,
                    validation_ratio=args.validation_ratio,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    l2=args.l2,
                    seed=args.seed,
                )
                OptimizerPipeline(config=config, logger=logger).run(
                    seed=args.seed,
                    candidate_pool=args.candidate_pool,
                    top_games=args.top_games,
                    generations=args.generations,
                    population=args.population,
                    draw_hour=args.draw_hour,
                    draw_minute=args.draw_minute,
                    engine=args.engine,
                    exhaustive_limit=args.exhaustive_limit,
                )
            elif args.mode == "experimental":
                OptimizerPipeline(config=config, logger=logger).run(
                    seed=args.seed,
                    candidate_pool=max(args.candidate_pool, 50000),
                    top_games=max(args.top_games, 200),
                    generations=max(args.generations, 50),
                    population=max(args.population, 150),
                    draw_hour=args.draw_hour,
                    draw_minute=args.draw_minute,
                    engine=args.engine,
                    exhaustive_limit=args.exhaustive_limit,
                )
            summary = PredictorPipeline(config=config, logger=logger).predict(
                seed=args.seed,
                candidate_pool=args.candidate_pool,
                top_games=args.top_games,
                generations=args.generations,
                population=args.population,
                max_overlap=args.max_overlap_final,
                draw_hour=args.draw_hour,
                draw_minute=args.draw_minute,
                engine=args.engine,
                exhaustive_limit=args.exhaustive_limit,
            )
        elif args.optimize:
            summary = OptimizerPipeline(config=config, logger=logger).run(
                seed=args.seed,
                candidate_pool=args.candidate_pool,
                top_games=args.top_games,
                generations=args.generations,
                population=args.population,
                draw_hour=args.draw_hour,
                draw_minute=args.draw_minute,
                engine=args.engine,
                exhaustive_limit=args.exhaustive_limit,
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
        elif args.transitions:
            summary = TransitionPipeline(config=config, logger=logger).run()
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
