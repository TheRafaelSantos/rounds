# Fase 0 - Revisao de Escopo e Arquitetura

## Resumo

O briefing pede um projeto novo, com foco atual 100% Lotofacil. O codigo antigo deste repositorio e voltado a Mega-Sena, entao a decisao tecnica adotada foi criar um modulo novo e isolado em `src/lotofacil_analytics`, sem apagar o material anterior.

## Fonte de dados escolhida

Fonte primaria: endpoint publico da CAIXA:

`https://servicebus2.caixa.gov.br/portaldeloterias/api/lotofacil`

O endpoint sem numero retorna o concurso mais recente. O endpoint com numero retorna um concurso especifico:

`https://servicebus2.caixa.gov.br/portaldeloterias/api/lotofacil/{numero}`

## Riscos da fonte

1. A API e publica, mas nao foi tratada aqui como contrato formal estavel.
2. O formato pode mudar sem aviso.
3. A rede pode falhar ou responder com lentidao.
4. Campos historicos podem vir nulos ou com nomes diferentes.

Mitigacoes implementadas na Fase 1:

1. Timeout por requisicao.
2. Tentativas automaticas em erro temporario.
3. Salvamento do JSON bruto por concurso.
4. Normalizacao separada da coleta.
5. Validacao de dezenas, datas, duplicidade e continuidade do historico.
6. Logs em arquivo local.

## Arquitetura da Fase 1

Entrada principal:

`main.py`

Pacote novo:

`src/lotofacil_analytics`

Responsabilidades:

1. `caixa_client.py`: acesso HTTP ao endpoint da CAIXA.
2. `normalize.py`: transformacao do JSON bruto em registro tabular.
3. `validators.py`: validacoes obrigatorias da Fase 1.
4. `storage.py`: persistencia de JSON bruto, CSV, Excel e estado local.
5. `pipeline.py`: orquestracao da atualizacao completa ou incremental.
6. `logger.py`: logs em console e arquivo.
7. `config.py`: caminhos e configuracoes.

## Saidas locais

As saidas da Fase 1 sao geradas localmente e ignoradas pelo Git:

1. `data/raw/lotofacil/concurso_000001.json`
2. `data/processed/lotofacil_concursos.csv`
3. `data/processed/lotofacil_state.json`
4. `data/exports/lotofacil_historico.xlsx`
5. `logs/lotofacil_analytics.log`

## Inconsistencias tecnicas do briefing

1. O briefing cita muitas fases avancadas, mas tambem determina que a IA nao deve implementar tudo de uma vez. A execucao correta e faseada.
2. O projeto atual do repositorio e Mega-Sena; reutilizar diretamente os scripts antigos aumentaria risco de misturar regras de 6 dezenas com Lotofacil de 15 dezenas.
3. A exigencia de Excel estruturado nao deve significar versionar planilhas geradas no Git; a planilha deve ser reprodutivel por comando.
4. Machine learning e geradores finais so fazem sentido depois da base historica validada.

## Proximo passo tecnico

Depois da Fase 1 validada, a Fase 2 deve calcular features basicas sem vazamento de dados:

1. pares e impares;
2. soma;
3. faixas 1-5, 6-10, 11-15, 16-20, 21-25;
4. linhas e colunas do volante;
5. gaps e sequencias;
6. repeticao em relacao ao concurso anterior.

## Fase 2 implementada

A Fase 2 gera `data/processed/lotofacil_features_base.csv` e `data/exports/lotofacil_features_base.xlsx`.

As features foram mantidas fora do arquivo base `lotofacil_concursos.csv` para preservar a camada de dados normalizados da Fase 1. O comando e:

`python main.py --features`

## Fase 3 implementada

A Fase 3 gera historico por dezena com o comando:

`python main.py --dezenas`

Saidas:

1. `data/processed/lotofacil_dezenas_long.csv`
2. `data/processed/lotofacil_dezenas_historico.csv`
3. `data/exports/lotofacil_dezenas_historico.xlsx`

A tabela `dezenas_historico` tem 25 linhas por concurso e calcula frequencia, atraso e rankings apenas com concursos anteriores.

## Fase 4 implementada

A Fase 4 gera combinacoes e assinaturas com o comando:

`python main.py --combinacoes`

Saidas:

1. `data/processed/lotofacil_combinacoes_features.csv`
2. `data/processed/lotofacil_combinacoes_pares.csv`
3. `data/processed/lotofacil_combinacoes_trios.csv`
4. `data/processed/lotofacil_combinacoes_quartetos.csv`
5. `data/exports/lotofacil_combinacoes.xlsx`

A tabela de features por concurso calcula frequencias contra o historico anterior. Os agregados de pares, trios e quartetos usam todo o historico e devem ser lidos como auditoria exploratoria.
