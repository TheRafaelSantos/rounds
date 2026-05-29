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

O codigo antigo de Mega-Sena foi preservado. A implementacao nova da Lotofacil fica isolada em:

`src/lotofacil_analytics`

## Fonte de dados

Endpoint publico da CAIXA:

`https://servicebus2.caixa.gov.br/portaldeloterias/api/lotofacil`

Risco: o endpoint e publico, mas pode mudar formato ou ficar indisponivel. Por isso o sistema salva JSON bruto, usa timeout, retentativas, logs e validacoes.

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

Gerar combinacoes e assinaturas:

```powershell
python main.py --combinacoes
```

Rodar backtest inicial:

```powershell
python main.py --backtest
```

Rodar backtest com parametros:

```powershell
python main.py --backtest --n-eval 500 --min-history 300 --seed 123 --window 100 --candidates 2000
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

Gerar jogos por metodo especifico:

```powershell
python main.py --generate-games --method balanceado_basico --qty 10
```

Gerar exatamente 2 jogos finais:

```powershell
python main.py --predict
```

Gerar os 2 jogos finais em modo completo, recalculando as fases antes da selecao:

```powershell
python main.py --predict --mode completo
```

Gerar Excel consolidado com as abas do briefing:

```powershell
python main.py --export
```

Abrir interface web local:

```powershell
python main.py --serve
```

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
data/processed/lotofacil_backtest.csv
data/processed/lotofacil_backtest_summary.csv
data/processed/lotofacil_auditoria_resumo.csv
data/processed/lotofacil_auditoria_dezenas.csv
data/processed/lotofacil_auditoria_anomalias.csv
data/processed/lotofacil_auditoria_monte_carlo.csv
data/processed/lotofacil_ml_dataset.csv
data/processed/lotofacil_ml_predictions.csv
data/processed/lotofacil_ml_summary.csv
data/processed/lotofacil_optimizer_candidates.csv
data/processed/lotofacil_optimizer_summary.csv
data/processed/lotofacil_jogos_gerados.csv
data/processed/lotofacil_prediction.csv
data/processed/lotofacil_state.json
data/exports/lotofacil_historico.xlsx
data/exports/lotofacil_features_base.xlsx
data/exports/lotofacil_dezenas_historico.xlsx
data/exports/lotofacil_combinacoes.xlsx
data/exports/lotofacil_backtest.xlsx
data/exports/lotofacil_auditoria.xlsx
data/exports/lotofacil_ml.xlsx
data/exports/lotofacil_optimizer.xlsx
data/exports/lotofacil_jogos_gerados.xlsx
data/exports/lotofacil_prediction.xlsx
data/exports/lotofacil_prediction_report.md
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

Componentes do score:

1. equilibrio estatistico: soma, pares, faixas, repeticao com ultimo concurso e sequencias;
2. historico recente: media de frequencia nos ultimos 100 concursos;
3. anti-popularidade humana: penaliza excesso de numeros baixos, linhas/colunas completas, diagonais fortes e sequencias longas;
4. combinatorio: penaliza pares historicamente muito frequentes;
5. contextual: considera data do proximo concurso, dia da semana, mes, trimestre, semestre, estacao do ano, fase da lua no horario de Brasilia e numerologia exploratoria.

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
6. diversidade minima configuravel por `--max-overlap-final`.

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
6. score contextual historico por dia da semana, periodo do ano e fase lunar.

A API da CAIXA nao informa a hora exata dentro do JSON historico. O sistema usa 20:00 no fuso `America/Sao_Paulo` como padrao configuravel:

```powershell
python main.py --predict --draw-hour 20 --draw-minute 0
```

Os dois jogos gerados sao sempre jogos completos de 15 dezenas. O sistema nao divide uma previsao em metades entre sugestoes.

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
12. `backtest`;
13. `jogos_gerados`;
14. `parametros`;
15. `logs_execucao`.

## Interface e executavel da Fase 10

Interface web local:

```powershell
python main.py --serve --host 127.0.0.1 --port 8765
```

Depois abra no navegador:

```text
http://127.0.0.1:8765
```

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
