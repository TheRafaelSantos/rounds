# Lotofacil Analytics

Projeto tecnico, educacional e estatistico para estudar historico da Lotofacil com Python.

Este projeto nao promete previsao de resultado, nao incentiva aposta e nao deve ser usado como garantia de ganho. A proposta e coletar dados, validar historico, gerar estudos e comparar qualquer metodo futuro contra baseline aleatorio.

## Estado atual

Fases implementadas:

1. **Fase 1 - Base de dados**.
2. **Fase 2 - Features basicas**.
3. **Fase 3 - Historico por dezena**.
4. **Fase 4 - Combinacoes e assinaturas**.
5. **Fase 5 - Backtesting inicial**.
6. **Fase 6 - Auditoria estatistica exploratoria**.
7. **Fase 7 - Machine learning temporal leve**.
8. **Fase 8 - Otimizacao heuristica de candidatos**.
9. **Fase 9 - Geracao final de 2 jogos**.
10. **Fase 10 - Interface local e build de executavel**.
11. **Camada superior - jogo unico, validacao exaustiva, ablation test e ajuste de pesos**.
12. **Camada Mandel/bolao - universo recomendado, desdobramento completo e fechamento reduzido**.
13. **Camada climatica experimental - temperatura, sensacao termica, umidade, pressao, chuva, anomalia e faixas climaticas**.
14. **Temporal profundo - dia da semana, ultimos 15/30 dias, bimestre, trimestre e semestre**.
15. **Calibracao tecnica walk-forward - estudo de pesos por concursos passados, sem comandar a geracao oficial**.
16. **Aprendizado supervisionado - usa o gabarito dos concursos historicos para calibrar a contribuicao media de cada estudo e gravar esses pesos no motor principal**.

Tambem estao implementados: analise pos-sorteio, auditoria de falsos negativos/falsos positivos, backtest especifico do score final `ensemble_score_v2` contra baseline aleatorio, motor exaustivo `ensemble_score_v4_exaustivo_transicao`, camada climatica e camada de decisao acima do motor atual.

O codigo antigo de Mega-Sena foi preservado. A implementacao nova da Lotofacil fica isolada em:

`src/lotofacil_analytics`

## Fonte de dados

Endpoint publico da CAIXA:

`https://servicebus2.caixa.gov.br/portaldeloterias/api/lotofacil`

Risco: o endpoint e publico, mas pode mudar formato ou ficar indisponivel. Por isso o sistema salva JSON bruto, usa timeout, retentativas, logs e validacoes.

Fonte climatica usada na camada experimental:

`https://open-meteo.com/en/docs/historical-weather-api`

O clima e consultado por cidade/UF do sorteio, horario assumido de Brasilia e cache local. Essa camada e testavel, mas tem plausibilidade baixa como causa fisica direta do resultado.

## Instalar no Windows

Abra o PowerShell na pasta do projeto:

```powershell
cd C:\Users\rafap\Desktop\LotoFacil
```

Crie e ative um ambiente virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instale dependencias:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Atualizar historico

Primeira execucao ou atualizacao incremental:

```powershell
python main.py --update
```

Rebaixar tudo desde o concurso 1:

```powershell
python main.py --full
```

Ver estado local:

```powershell
python main.py --status
```

Gerar features basicas:

```powershell
python main.py --features
```

Gerar historico por dezena:

```powershell
python main.py --dezenas
```

Gerar camada climatica historica:

```powershell
python main.py --climate --draw-hour 20 --draw-minute 0
```

Para testar sem processar todas as cidades/UF de uma vez:

```powershell
python main.py --climate --climate-max-locations 2 --draw-hour 20 --draw-minute 0
```

Arquivos gerados:

1. `data/processed/lotofacil_clima.csv`;
2. `data/processed/lotofacil_clima_summary.csv`;
3. `data/exports/lotofacil_clima.xlsx`;
4. cache Open-Meteo em `data/raw/climate_open_meteo`.

Gerar temporal profundo incremental:

```powershell
python main.py --temporal-deep
```

Arquivos gerados:

1. `data/processed/lotofacil_temporal_profundo.csv`;
2. `data/processed/lotofacil_temporal_profundo_summary.csv`;
3. `data/exports/lotofacil_temporal_profundo.xlsx`.

Rodar calibracao tecnica walk-forward:

```powershell
python main.py --calibrate-engine --calibration-from-concurso 2500 --calibration-baseline-samples 30 --draw-hour 20 --draw-minute 0
```

Arquivos gerados:

1. `data/processed/lotofacil_engine_calibration.csv`;
2. `data/processed/lotofacil_engine_calibration_summary.csv`;
3. `data/processed/lotofacil_engine_calibrated_weights.json`;
4. `data/exports/lotofacil_engine_calibration.xlsx`.

Esse comando fica como estudo tecnico. A geracao oficial dos 2 jogos usa somente os pesos do aprendizado supervisionado.

Rodar piloto retomavel de busca de pesos por resultado final:

```powershell
python main.py --calibration-pilot --pilot-concurso 2500 --pilot-games 100 --pilot-candidate-pool 5000 --draw-hour 20 --draw-minute 0
```

Esse piloto nao altera os pesos oficiais do motor principal. Ele cria uma base de candidatos para o concurso-alvo, testa combinacoes de pesos, gera 2 jogos por tentativa e compara somente a quantidade de acertos. Se o computador desligar, rode o mesmo comando novamente e ele continua das tentativas ja salvas.

Para reiniciar do zero o piloto de um concurso:

```powershell
python main.py --calibration-pilot --pilot-concurso 2500 --pilot-games 100 --pilot-reset --draw-hour 20 --draw-minute 0
```

Arquivos gerados para o concurso 2500:

1. `data/processed/lotofacil_calibration_pilot_candidates_2500.csv`;
2. `data/processed/lotofacil_calibration_pilot_results_2500.csv`;
3. `data/processed/lotofacil_calibration_pilot_summary_2500.csv`;
4. `data/processed/lotofacil_calibration_pilot_state_2500.json`;
5. `data/exports/lotofacil_calibration_pilot_2500.xlsx`.

Rodar aprendizado supervisionado com gabarito historico:

```powershell
python main.py --supervised-calibration --supervised-from-concurso 2500 --supervised-samples 800 --supervised-max-contests 25 --draw-hour 20 --draw-minute 0
```

Esse comando usa concursos ja encerrados como gabarito. Para cada concurso historico, ele calcula os scores de todos os estudos para a sequencia real e para uma amostra de combinacoes concorrentes. Depois mede quais estudos colocariam a sequencia real em melhor posicao, salva os pesos aprendidos e atualiza a media em `lotofacil_supervised_calibrated_weights.json`. O motor principal passa a carregar somente esse arquivo ao gerar novos jogos.

Para rodar continuamente e retomar de onde parou:

```powershell
python main.py --supervised-calibration --supervised-loop --supervised-from-concurso 2500 --supervised-samples 800 --supervised-max-contests 25 --supervised-sleep-seconds 30 --draw-hour 20 --draw-minute 0
```

Arquivos principais:

1. `data/processed/lotofacil_supervised_calibration_state.json`;
2. `data/processed/lotofacil_supervised_calibration_results.csv`;
3. `data/processed/lotofacil_supervised_calibration_summary.csv`;
4. `data/processed/lotofacil_supervised_calibration_weights.csv`;
5. `data/processed/lotofacil_supervised_calibrated_weights.json`;
6. `data/exports/lotofacil_supervised_calibration.xlsx`.

Na interface web, clique em **Aprendizado supervisionado** para acompanhar status, pesos atuais, ranking antes/depois e ultimos concursos aprendidos.

Gerar combinacoes e assinaturas:

```powershell
python main.py --combinacoes
```

Analisar transicoes entre concursos consecutivos:

```powershell
python main.py --transitions
```

Rodar backtest inicial:

```powershell
python main.py --backtest
```

Rodar backtest com parametros:

```powershell
python main.py --backtest --n-eval 500 --min-history 300 --seed 123 --window 100 --candidates 2000
```

Rodar backtest do score final completo:

```powershell
python main.py --final-backtest --final-n-eval 60 --min-history 300 --final-candidate-pool 2500 --final-generations 6 --final-population 40 --top-games 100 --max-overlap-final 8
```

Rodar auditoria estatistica:

```powershell
python main.py --audit
```

Rodar auditoria com mais simulacoes Monte Carlo:

```powershell
python main.py --audit --monte-carlo-runs 2000 --seed 123
```

Rodar ML temporal:

```powershell
python main.py --ml
```

Rodar ML com parametros:

```powershell
python main.py --ml --train-ratio 0.70 --validation-ratio 0.15 --epochs 400 --learning-rate 0.05 --l2 0.001 --seed 123
```

Gerar candidatos otimizados:

```powershell
python main.py --optimize
```

Gerar candidatos com parametros:

```powershell
python main.py --optimize --candidate-pool 20000 --top-games 100 --generations 30 --population 100 --seed 123
```

Gerar candidatos avaliando todas as 3.268.760 combinacoes possiveis:

```powershell
python main.py --optimize --engine exaustivo --top-games 5000 --draw-hour 20 --draw-minute 0
```

Gerar jogos por metodo especifico:

```powershell
python main.py --generate-games --method balanceado_basico --qty 10
```

Gerar exatamente 2 jogos finais:

```powershell
python main.py --predict
```

Gerar os 2 jogos finais com o motor exaustivo:

```powershell
python main.py --predict --engine exaustivo --draw-hour 20 --draw-minute 0
```

Gerar os 2 jogos finais em modo completo, recalculando as fases antes da selecao:

```powershell
python main.py --predict --mode completo
```

Gerar 1 jogo unico pela camada superior de decisao:

```powershell
python main.py --predict-single --engine exaustivo --draw-hour 20 --draw-minute 0
```

Gerar plano Mandel/bolao com desdobramento e fechamento reduzido:

```powershell
python main.py --mandel --mandel-universe-size 18 --mandel-guarantee-hits 14 --mandel-max-reduced-games 80 --draw-hour 20 --draw-minute 0
```

Backtest walk-forward do jogo unico exaustivo:

```powershell
python main.py --backtest-exhaustive --validation-n-eval 3 --min-history 300 --draw-hour 20 --draw-minute 0
```

Teste de ablation, removendo uma familia de score por vez:

```powershell
python main.py --ablation-test --validation-n-eval 3 --min-history 300 --draw-hour 20 --draw-minute 0
```

Ajuste comparativo de perfis de pesos:

```powershell
python main.py --tune-weights --validation-n-eval 3 --min-history 300 --draw-hour 20 --draw-minute 0
```

Analisar um resultado real contra os jogos previstos:

```powershell
python main.py --analyze-result --result-label sexta_2026-05-29 --actual-numbers "01 03 05 06 07 08 09 10 12 13 16 18 20 21 23"
```

Gerar Excel consolidado com as abas do briefing:

```powershell
python main.py --export
```

Abrir interface web local:

```powershell
python main.py --serve
```

Rodar com Docker no Windows:

```powershell
cd C:\Users\rafap\Desktop\LotoFacil
docker compose up -d --build
```

Abrir a interface Docker:

```text
http://127.0.0.1:8765
```

Ver containers:

```powershell
docker compose ps
```

Ver logs do aprendizado supervisionado:

```powershell
docker compose logs -f --tail=120 lotofacil-supervised
```

Parar sem apagar dados:

```powershell
docker compose down
```

Na VPS/Linux, rode os mesmos comandos dentro de `/opt/lotofacil`. O guia de operacao fica em `docs/DEPLOY_VPS.md`.

Gerar executavel Windows, se PyInstaller estiver instalado:

```powershell
python main.py --build-exe
```

## Saidas geradas

Arquivos locais gerados:

```text
data/raw/lotofacil/
data/processed/lotofacil_concursos.csv
data/processed/lotofacil_features_base.csv
data/processed/lotofacil_dezenas_long.csv
data/processed/lotofacil_dezenas_historico.csv
data/processed/lotofacil_combinacoes_features.csv
data/processed/lotofacil_combinacoes_pares.csv
data/processed/lotofacil_combinacoes_trios.csv
data/processed/lotofacil_combinacoes_quartetos.csv
data/processed/lotofacil_transicoes.csv
data/processed/lotofacil_transicoes_summary.csv
data/processed/lotofacil_transicoes_dezenas.csv
data/processed/lotofacil_backtest.csv
data/processed/lotofacil_backtest_summary.csv
data/processed/lotofacil_backtest_final_score.csv
data/processed/lotofacil_backtest_final_score_summary.csv
data/processed/lotofacil_auditoria_resumo.csv
data/processed/lotofacil_auditoria_dezenas.csv
data/processed/lotofacil_auditoria_anomalias.csv
data/processed/lotofacil_auditoria_monte_carlo.csv
data/processed/lotofacil_ml_dataset.csv
data/processed/lotofacil_ml_predictions.csv
data/processed/lotofacil_ml_summary.csv
data/processed/lotofacil_optimizer_candidates.csv
data/processed/lotofacil_optimizer_summary.csv
data/processed/lotofacil_prediction_single.csv
data/processed/lotofacil_backtest_exaustivo_single.csv
data/processed/lotofacil_backtest_exaustivo_single_summary.csv
data/processed/lotofacil_ablation_test.csv
data/processed/lotofacil_ablation_test_summary.csv
data/processed/lotofacil_tune_weights_results.csv
data/processed/lotofacil_tune_weights_summary.csv
data/processed/lotofacil_tuned_weights.json
data/processed/lotofacil_mandel_plan.csv
data/processed/lotofacil_mandel_games.csv
data/processed/lotofacil_jogos_gerados.csv
data/processed/lotofacil_prediction.csv
data/processed/lotofacil_pos_sorteio_jogos_<rotulo>.csv
data/processed/lotofacil_pos_sorteio_dezenas_<rotulo>.csv
data/processed/lotofacil_state.json
data/exports/lotofacil_historico.xlsx
data/exports/lotofacil_features_base.xlsx
data/exports/lotofacil_dezenas_historico.xlsx
data/exports/lotofacil_combinacoes.xlsx
data/exports/lotofacil_backtest.xlsx
data/exports/lotofacil_backtest_final_score.xlsx
data/exports/lotofacil_auditoria.xlsx
data/exports/lotofacil_ml.xlsx
data/exports/lotofacil_optimizer.xlsx
data/exports/lotofacil_jogos_gerados.xlsx
data/exports/lotofacil_prediction.xlsx
data/exports/lotofacil_prediction_report.md
data/exports/lotofacil_mandel.xlsx
data/exports/lotofacil_mandel_report.md
data/exports/lotofacil_pos_sorteio_<rotulo>.xlsx
data/exports/lotofacil_pos_sorteio_report_<rotulo>.md
data/exports/lotofacil_analytics_completo.xlsx
logs/lotofacil_analytics.log
```

Esses arquivos sao ignorados pelo Git porque sao reproduziveis por comando e mudam conforme a CAIXA publica novos concursos.

## Validacoes da Fase 1

1. Cada concurso precisa ter exatamente 15 dezenas.
2. Todas as dezenas precisam estar entre 1 e 25.
3. Nao pode haver dezena repetida no mesmo concurso.
4. Concurso nao pode duplicar.
5. A data do sorteio precisa ser valida.
6. O historico local precisa ser continuo, sem buracos entre primeiro e ultimo concurso.

## Features da Fase 2

O comando `python main.py --features` gera uma tabela separada com:

1. features temporais: ano, mes, dia, semana do mes, quinzena, bimestre, trimestre e semestre;
2. paridade: quantidade de pares e impares;
3. soma, media, menor dezena, maior dezena e amplitude;
4. contagem de primos, Fibonacci e quadrados perfeitos;
5. faixas 01-05, 06-10, 11-15, 16-20 e 21-25;
6. linhas e colunas do volante 5x5;
7. gaps, sequencias consecutivas e maior sequencia;
8. repeticao em relacao ao concurso anterior.

Essas features nao usam concursos futuros. Frequencia historica, atraso e rankings dinamicos ficam para a Fase 3.

## Historico por dezena da Fase 3

O comando `python main.py --dezenas` gera:

1. `dezenas_long`: 15 linhas por concurso, uma para cada dezena sorteada;
2. `dezenas_historico`: 25 linhas por concurso, uma para cada dezena possivel da Lotofacil.

A tabela `dezenas_historico` calcula, antes de cada concurso:

1. frequencia total ate o concurso anterior;
2. frequencia nos ultimos 5, 10, 20, 50 e 100 concursos;
3. atraso atual;
4. media de atraso historico;
5. rankings por frequencia total, frequencia recente e atraso;
6. flags estaticas da dezena, como par, prima, Fibonacci, linha e coluna do volante.

O alvo `saiu_no_concurso` indica se a dezena saiu naquele concurso. As features historicas dessa linha usam apenas concursos anteriores.

## Combinacoes e assinaturas da Fase 4

O comando `python main.py --combinacoes` gera:

1. features combinatorias por concurso;
2. ranking historico de todos os pares possiveis;
3. ranking historico de todos os trios possiveis;
4. ranking historico de todos os quartetos possiveis;
5. assinaturas de paridade, faixas, linhas, colunas, gaps, modulo 3, modulo 5 e repeticao anterior.

Na tabela por concurso, as frequencias de pares, trios e quartetos sao calculadas somente contra concursos anteriores. Os rankings agregados usam todo o historico e servem para auditoria exploratoria, nao para previsao direta.

## Backtesting da Fase 5

O comando `python main.py --backtest` executa backtest walk-forward nos concursos finais da base.

Metodos iniciais:

1. `aleatorio_puro`;
2. `frequencia_quente`;
3. `frequencia_fria`;
4. `hibrido_quente_frio`;
5. `balanceado_basico`.

O treino de cada linha usa somente concursos anteriores ao concurso avaliado. O resultado mede acertos de 11, 12, 13, 14 e 15 dezenas, alem da media de acertos por metodo.

Esses metodos sao baselines e heuristicas simples. Resultado melhor em uma janela nao prova previsao real; serve para comparar contra o acaso e detectar overfitting nas fases futuras.

## Backtesting do score final

O comando `python main.py --final-backtest` testa o metodo final `ensemble_score_v2` em modo walk-forward.

Ele simula concursos passados assim:

1. separa apenas o historico anterior ao concurso avaliado;
2. gera candidatos com o mesmo otimizador usado na previsao final;
3. seleciona 2 jogos completos com limite de sobreposicao;
4. compara contra o resultado real;
5. compara o desempenho contra `baseline_2_jogos_aleatorios`.

Saidas:

1. `data/processed/lotofacil_backtest_final_score.csv`;
2. `data/processed/lotofacil_backtest_final_score_summary.csv`;
3. `data/exports/lotofacil_backtest_final_score.xlsx`.

## Auditoria estatistica da Fase 6

O comando `python main.py --audit` gera uma auditoria exploratoria com:

1. frequencia observada versus frequencia esperada por dezena;
2. qui-quadrado aproximado para frequencia marginal das dezenas;
3. entropia das dezenas;
4. anomalias simples de soma, paridade e repeticao em relacao ao concurso anterior;
5. comparacao Monte Carlo da repeticao media entre concursos.

O p-value do qui-quadrado usa aproximacao Wilson-Hilferty para evitar dependencia pesada nesta fase. A comparacao de repeticao usa p-value empirico por simulacao.

Auditoria nao e previsao. Ela serve para verificar se o historico parece compativel com aleatoriedade e para apontar casos extremos para estudo.

## Machine learning da Fase 7

O comando `python main.py --ml` cria um dataset temporal com uma linha por concurso e por dezena, treina uma regressao logistica simples implementada com `numpy` e compara contra:

1. `baseline_freq_100`;
2. `baseline_atraso`;
3. `baseline_random`.

O split e temporal. O modelo treina nos concursos iniciais, valida no bloco seguinte e testa no bloco final. As features de cada linha usam apenas concursos anteriores ao concurso avaliado.

Esta fase nao usa `scikit-learn` para manter instalacao leve. O objetivo e criar uma referencia auditavel, nao maximizar complexidade.

## Otimizacao da Fase 8

O comando `python main.py --optimize` gera candidatos ranqueados por score composto.

Por padrao, o motor atual e `ensemble_score_v4_exaustivo_transicao`. Ele avalia todas as `3.268.760` combinacoes possiveis da Lotofacil antes de salvar os melhores candidatos. Se precisar voltar ao motor anterior, use:

```powershell
python main.py --optimize --engine heuristico
```

Componentes do score:

1. equilibrio estatistico: soma, pares, faixas, repeticao com ultimo concurso e sequencias, usando bandas historicas em vez de alvo fixo;
2. historico recente: media de frequencia nos ultimos 100 concursos;
3. anti-popularidade humana: penaliza excesso de padroes populares, mas sem bloquear linhas, colunas, diagonais e sequencias quando elas aparecem como cenario historico plausivel;
4. combinatorio: penaliza pares historicamente muito frequentes;
5. contextual: considera data do proximo concurso, dia da semana, mes, trimestre, semestre, estacao do ano, fase da lua no horario de Brasilia e numerologia exploratoria;
6. climatico experimental: temperatura, sensacao termica, umidade relativa, pressao atmosferica, chuva, anomalia de temperatura e faixas climaticas do local/horario do sorteio;
7. temporal profundo: dia da semana dinamico, ultimos 15/30 dias, bimestre, trimestre e semestre;
8. cenarios: reforca soma baixa/media/alta, sequencia forte, visual forte, assinatura historica por faixas e faixa alta 21-25;
9. contrarian: monitora dezenas que os jogos anteriores penalizaram demais, como 01, 13 e 22;
10. transicao: compara cada concurso com o concurso seguinte e aprende repetidas, entradas, saidas, delta de soma e mudanca por faixas.

O score final `ensemble_score_v2` combina esses blocos com pesos documentados na aba/CSV de resumo do otimizador. A geracao de candidatos usa perfis variados, incluindo `soma_baixa`, `sequencia_forte`, `visual_forte`, `contrarian_controlado`, `faixa_alta_reforcada`, `assinatura_historica`, `weighted_temporal`, `monte_carlo_filtrado` e `genetico_simples`.

No `ensemble_score_v4_exaustivo_transicao`, tambem entram:

1. varredura completa de todas as combinacoes de 15 dezenas entre 25;
2. fase da lua calculada para todos os concursos historicos com horario de Brasilia configuravel;
3. numerologia da data, do concurso, do dia+mes e das dezenas;
4. localidade historica por local, cidade e UF;
5. bairro somente se existir coluna confiavel na base. Na base atual da CAIXA, bairro nao esta disponivel;
6. clima historico do local do sorteio, quando `python main.py --climate` ja foi executado;
7. transicao sequencial `concurso N -> concurso N+1`, incluindo dezenas que ficaram, entraram, sairam e continuaram fora.

Essa fase gera candidatos para a selecao final. Ela nao afirma que os candidatos sao previsoes garantidas.

## Predicao final da Fase 9

O comando `python main.py --predict` mostra exatamente 2 jogos finais de 15 dezenas.

Ele usa o ranking gerado pelo otimizador quando disponivel. Se o ranking ainda nao existir, gera candidatos internamente antes da selecao.

Regras:

1. exatamente 2 jogos;
2. 15 dezenas por jogo;
3. dezenas entre 1 e 25;
4. sem repeticao dentro do mesmo jogo;
5. jogos distintos;
6. diversidade minima configuravel por `--max-overlap-final`, com padrao 8;
7. Jogo 1 vem do melhor candidato do motor principal;
8. Jogo 2 passa por um seletor inteligente de portfolio: entre os candidatos que respeitam a diversidade, ele combina `score_final`, `score_transicao`, `score_contextual`, forca dos componentes e dezenas exclusivas contra o Jogo 1.

A saida de tela e curta. O detalhe tecnico fica em `data/exports/lotofacil_prediction_report.md`.

Modos disponiveis:

1. `--mode rapido`: usa os arquivos ja calculados e gera os 2 jogos rapidamente.
2. `--mode completo`: atualiza a base, recalcula features, dezenas, combinacoes, backtest, auditoria, ML, otimizacao e depois gera os 2 jogos.
3. `--mode experimental`: refaz a otimizacao com limites maiores e depois gera os 2 jogos.

Contexto usado na previsao final:

1. `data_proximo_concurso` vinda da API da CAIXA;
2. dia da semana dessa data;
3. periodo do ano: mes, trimestre, semestre e estacao;
4. fase, idade e iluminacao aproximada da lua no horario de Brasilia;
5. numerologia exploratoria da data, do concurso e de dia+mes;
6. score contextual historico por dia da semana, periodo do ano e fase lunar;
7. localidade historica do sorteio: local, cidade e UF quando informados pela API da CAIXA;
8. clima do local/horario assumido do sorteio: temperatura, sensacao termica, umidade, pressao, chuva, anomalia contra media recente e faixas climaticas.

A API da CAIXA nao informa a hora exata dentro do JSON historico. O sistema usa 20:00 no fuso `America/Sao_Paulo` como padrao configuravel:

```powershell
python main.py --predict --draw-hour 20 --draw-minute 0
```

Os dois jogos gerados sao sempre jogos completos de 15 dezenas. O sistema nao divide uma previsao em metades entre sugestoes.

## Camada superior de decisao

Essa camada nao substitui as analises existentes. Ela fica por cima do motor `ensemble_score_v4_exaustivo_transicao` e usa os mesmos blocos: estatistica, historico, atrasos, combinacoes, lua, dia da semana, periodo do ano, numerologia, localidade, clima, cenarios, contrarian e transicao sequencial.

Na selecao final, a camada superior agora aplica uma decisao protegida:

1. `score_decisao_protegida`: combina o score final do motor com transicao, contexto protegido, consenso do top de candidatos e cobertura anti-falso-negativo;
2. `score_contexto_protegido`: lua, numerologia, dia da semana e localidade continuam entrando, mas viram bonus de ajuste fino; eles nao devem derrubar sozinhos um candidato forte por outros sinais;
3. `score_cobertura_risco_falso_negativo`: mede quantas dezenas menos consensuais, mas presentes em candidatos fortes, foram preservadas no jogo;
4. `dezenas_risco_falso_negativo`: lista as dezenas protegidas para reduzir o risco de ficarem fora do jogo unico e dos jogos finais;
5. `dezenas_nucleo_consenso`: mostra o nucleo que aparece com mais forca no top de candidatos.
6. `score_climatico`: mede a aderencia do candidato ao historico de dezenas em condicoes climaticas parecidas.

Comandos principais:

1. `python main.py --predict-single`: escolhe apenas o jogo mais bem ranqueado, como um unico conjunto completo de 15 dezenas.
2. `python main.py --backtest-exhaustive`: refaz a selecao jogo a jogo no passado, usando apenas concursos anteriores ao sorteio avaliado.
3. `python main.py --ablation-test`: remove uma familia de score por vez para medir se ela melhorou ou piorou o historico testado.
4. `python main.py --tune-weights`: compara perfis de pesos e salva o melhor perfil observado em `data/processed/lotofacil_tuned_weights.json`.

Perfis de pesos aceitos em `--weight-profile`:

1. `padrao_atual`;
2. `contexto_forte`;
3. `historico_forte`;
4. `combinatorio_forte`;
5. `contrarian_forte`;
6. `estatistico_forte`;
7. `cenarios_forte`;
8. `atraso_forte`;
9. `transicao_forte`.

Exemplo com perfil contextual mais forte:

```powershell
python main.py --predict-single --weight-profile contexto_forte --engine exaustivo --draw-hour 20 --draw-minute 0
```

Para testes rapidos de validacao tecnica, use `--exhaustive-limit`. Para varredura completa, omita esse parametro.

```powershell
python main.py --backtest-exhaustive --validation-n-eval 1 --min-history 300 --exhaustive-limit 50000
```

## Camada Mandel / bolao

Essa camada aplica no projeto a parte util e auditavel do metodo associado a Stefan Mandel: cobertura combinatoria. Ela nao tenta criar uma "formula magica"; ela organiza custo, universo de dezenas e quantidade de jogos para aumentar cobertura quando voce pretende jogar em grupo.

O sistema faz assim:

1. usa o motor exaustivo atual para escolher um universo recomendado de 15 a 20 dezenas;
2. calcula o desdobramento completo desse universo;
3. calcula o custo teorico por quantidade de jogos, usando R$ 3,50 por aposta simples;
4. monta um fechamento reduzido guloso para cobrir o maximo possivel de cenarios dentro do universo;
5. salva CSV, Markdown e Excel para conferencia.

Comando recomendado:

```powershell
python main.py --mandel --mandel-universe-size 18 --mandel-guarantee-hits 14 --mandel-max-reduced-games 80 --draw-hour 20 --draw-minute 0
```

Saidas:

1. `data/processed/lotofacil_mandel_plan.csv`: tabela de custo para universos de 15 a 20 dezenas;
2. `data/processed/lotofacil_mandel_games.csv`: jogos do fechamento reduzido;
3. `data/exports/lotofacil_mandel_report.md`: explicacao tecnica;
4. `data/exports/lotofacil_mandel.xlsx`: Excel com plano, jogos e dezenas protegidas.

Limite importante: o desdobramento completo de 18 dezenas gera 816 jogos. Ele so garante 15 pontos se as 15 dezenas sorteadas estiverem dentro das 18 dezenas escolhidas. O fechamento reduzido troca custo menor por garantia menor ou condicional.

Comprar todas as combinacoes da Lotofacil exigiria 3.268.760 jogos. Com aposta simples de R$ 3,50, o custo teorico seria R$ 11.440.660,00, fora logistica, limite operacional, rateio de premio e risco de nao haver retorno financeiro.

## Analise de transicoes

O comando `python main.py --transitions` compara o concurso 1 contra o 2, o 2 contra o 3, e assim por diante ate o concurso mais recente.

Ele salva:

1. `data/processed/lotofacil_transicoes.csv`: uma linha para cada par consecutivo;
2. `data/processed/lotofacil_transicoes_summary.csv`: resumo de repetidas, entradas, saidas, delta de soma e assinaturas;
3. `data/processed/lotofacil_transicoes_dezenas.csv`: probabilidade suavizada de cada dezena ficar quando saiu no concurso anterior ou entrar quando ficou fora;
4. `data/exports/lotofacil_transicoes.xlsx`: Excel com as tres abas.

O motor exaustivo usa esses dados no `score_transicao`. Para cada candidato ele compara o jogo proposto com o ultimo concurso disponivel e pontua:

1. quantidade de dezenas repetidas;
2. dezenas novas que entram;
3. dezenas que saem;
4. mudanca por faixas 01-05, 06-10, 11-15, 16-20 e 21-25;
5. delta de soma em relacao ao concurso anterior;
6. probabilidade historica de cada dezena permanecer ou entrar.

## Analise pos-sorteio

O comando `python main.py --analyze-result` compara um resultado real contra os jogos salvos em `data/processed/lotofacil_prediction.csv`.

Exemplo para o sorteio de sexta:

```powershell
python main.py --analyze-result --result-label sexta_2026-05-29 --actual-numbers "01 03 05 06 07 08 09 10 12 13 16 18 20 21 23"
```

Exemplo para o sorteio de sabado:

```powershell
python main.py --analyze-result --result-label sabado_2026-05-30 --actual-numbers "01 02 03 05 06 08 09 11 14 18 20 21 22 24 25"
```

Cada rótulo gera arquivos separados, evitando sobrescrever a auditoria anterior:

1. `data/processed/lotofacil_pos_sorteio_jogos_<rotulo>.csv`;
2. `data/processed/lotofacil_pos_sorteio_dezenas_<rotulo>.csv`;
3. `data/exports/lotofacil_pos_sorteio_report_<rotulo>.md`;
4. `data/exports/lotofacil_pos_sorteio_<rotulo>.xlsx`.

Essa analise mostra acertos por jogo, cobertura da uniao dos 2 jogos, dezenas sorteadas que ficaram fora, falsos positivos e hipoteses tecnicas para recalibracao.

## Geracao manual de jogos

O comando `python main.py --generate-games --method balanceado_basico --qty 10` gera jogos intermediarios para estudo e salva:

1. `data/processed/lotofacil_jogos_gerados.csv`;
2. `data/exports/lotofacil_jogos_gerados.xlsx`.

Metodos aceitos:

1. `aleatorio_puro`;
2. `balanceado_basico`;
3. `frequencia_quente`;
4. `frequencia_fria`;
5. `hibrido_quente_frio`;
6. `score_equilibrado`;
7. `anti_popularidade_humana`;
8. `monte_carlo_filtrado`;
9. `genetico_opcional`.

## Export consolidado

O comando `python main.py --export` gera `data/exports/lotofacil_analytics_completo.xlsx` com as abas do briefing:

1. `concursos_raw`;
2. `concursos`;
3. `concursos_features`;
4. `dezenas_long`;
5. `dezenas_historico`;
6. `frequencias`;
7. `atrasos`;
8. `pares`;
9. `trios`;
10. `quartetos`;
11. `rankings`;
12. `transicoes`;
13. `transicoes_resumo`;
14. `transicoes_dezenas`;
15. `backtest`;
16. `backtest_score_final`;
17. `backtest_score_final_resumo`;
18. `jogo_unico`;
19. `backtest_exaustivo`;
20. `backtest_exaustivo_resumo`;
21. `ablation_test`;
22. `ablation_test_resumo`;
23. `tune_weights`;
24. `tune_weights_resumo`;
25. `jogos_gerados`;
26. `pos_sorteio_jogos`;
27. `pos_sorteio_dezenas`;
28. `contexto_proximo_concurso`;
29. `parametros`;
30. `logs_execucao`.

## Interface e executavel da Fase 10

Interface web local:

```powershell
python main.py --serve --host 127.0.0.1 --port 8765
```

Depois abra no navegador:

```text
http://127.0.0.1:8765
```

A tela local permite atualizar a base, analisar transicoes, gerar 2 jogos, gerar o jogo unico da camada superior, gerar o plano Mandel/bolao e baixar os relatorios correspondentes.

Ao gerar 2 jogos, a interface mostra a comparacao visual dos scores dos dois jogos. O Jogo 2 tambem exibe o `score_portfolio_jogo_2`, o overlap contra o Jogo 1 e as dezenas exclusivas usadas para diversificar a sugestao.

Ao gerar jogo unico ou 2 jogos, a interface tambem mostra `Decisao protegida`, `Contexto protegido` e `Anti-falso-negativo`. Esses campos explicam quando a camada final manteve uma dezena por consenso, transicao ou risco de exclusao, mesmo se lua/numerologia/dia/localidade nao favorecerem totalmente aquela combinacao.

Build de executavel:

```powershell
python -m pip install pyinstaller
python main.py --build-exe
```

O executavel sera gerado em `dist/`. A pasta `dist/` nao entra no Git.

## Testes

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests
```

## Limitacoes

1. O projeto nao prova previsao de loteria; ele testa hipoteses contra baseline aleatorio.
2. Lua, dia da semana, periodo do ano e numerologia entram como fatores exploratorios de score. Eles nao sao tratados como prova cientifica de previsao.
3. Clima, peso de bolas, maquina usada, desgaste e manutencao nao entram como fator preditivo enquanto nao houver fonte historica publica, confiavel e auditavel para cada concurso.
4. O custo teorico da aposta nao e inferido da API da CAIXA; no backtest, premio teorico vem das faixas do concurso, mas custo/ROI ficam nulos quando nao houver fonte local confiavel.
5. A regressao logistica simples foi priorizada por auditabilidade. Modelos mais pesados devem ser adicionados apenas se trouxerem ganho real fora da amostra.

## Expansao futura para Mega-Sena

Para expandir para Mega-Sena, nao reutilize diretamente as regras da Lotofacil. Crie um pacote separado com:

1. regra de 6 dezenas entre 1 e 60;
2. novo normalizador;
3. validadores proprios;
4. features e backtests ajustados ao espaco combinatorio da Mega-Sena;
5. comandos isolados para evitar misturar bases.
