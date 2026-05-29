# Lotofacil Analytics

Projeto tecnico, educacional e estatistico para estudar historico da Lotofacil com Python.

Este projeto nao promete previsao de resultado, nao incentiva aposta e nao deve ser usado como garantia de ganho. A proposta e coletar dados, validar historico, gerar estudos e comparar qualquer metodo futuro contra baseline aleatorio.

## Estado atual

Fases implementadas:

1. **Fase 1 - Base de dados**.
2. **Fase 2 - Features basicas**.

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

## Saidas geradas

Arquivos locais gerados:

```text
data/raw/lotofacil/
data/processed/lotofacil_concursos.csv
data/processed/lotofacil_features_base.csv
data/processed/lotofacil_state.json
data/exports/lotofacil_historico.xlsx
data/exports/lotofacil_features_base.xlsx
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

1. Historico por dezena.
2. Pares, trios, quartetos e assinaturas.
3. Backtesting com baseline aleatorio.
4. Auditoria estatistica.
5. Geracao final de exatamente 2 jogos de 15 dezenas.
