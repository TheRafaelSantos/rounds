# Lotofacil Analytics

Projeto tecnico, educacional e estatistico para estudar historico da Lotofacil com Python.

Este projeto nao promete previsao de resultado, nao incentiva aposta e nao deve ser usado como garantia de ganho. A proposta e coletar dados, validar historico, gerar estudos e comparar qualquer metodo futuro contra baseline aleatorio.

## Estado atual

Fases implementadas:

1. **Fase 1 - Base de dados**.
2. **Fase 2 - Features basicas**.
3. **Fase 3 - Historico por dezena**.
4. **Fase 4 - Combinacoes e assinaturas**.

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
data/processed/lotofacil_state.json
data/exports/lotofacil_historico.xlsx
data/exports/lotofacil_features_base.xlsx
data/exports/lotofacil_dezenas_historico.xlsx
data/exports/lotofacil_combinacoes.xlsx
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

## Testes

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests
```

## Limitacoes

1. A Fase 1 ainda nao calcula features estatisticas.
2. Ainda nao ha backtesting.
3. Ainda nao ha geracao dos 2 jogos finais.
4. Machine learning fica para fases posteriores, depois da base validada.

## Proximas fases

1. Backtesting com baseline aleatorio.
2. Auditoria estatistica.
3. Geracao final de exatamente 2 jogos de 15 dezenas.
