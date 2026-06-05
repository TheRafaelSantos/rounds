# Deploy na VPS - Lotofacil Analytics

## Resumo

O projeto roda em dois containers Docker:

1. `lotofacil-web`: interface web na porta `8765`.
2. `lotofacil-calibrator`: calibrador 24/7 que tenta pesos diferentes ate acertar 15 dezenas em um dos dois jogos historicos.

Os arquivos de progresso ficam no volume `./data`, fora da imagem Docker. Se o container reiniciar, o processo continua do estado salvo.

A calibracao usa cache por concurso em `data/processed/lotofacil_calibration_lab_cache`. A primeira tentativa de um concurso cria a matriz de scores; as tentativas seguintes reaproveitam essa matriz e apenas recalculam pesos.

Tentativas com 11, 12, 13 ou 14 acertos ficam salvas em `data/processed/lotofacil_calibration_lab_elites.csv`. O calibrador usa esses registros como memoria local para mutar, cruzar e refinar pesos que ja chegaram perto. Quando encontra 15 acertos em um dos dois jogos, salva em `lotofacil_calibration_lab_winners.csv`, recalcula a media vencedora e avanca para o proximo concurso.

O calibrador tambem aplica penalizacao anti-repeticao: jogos ja testados muitas vezes, ou quase iguais aos jogos recentes, perdem score na tentativa atual. Isso reduz desperdicio quando a busca entra em plato repetindo a mesma combinacao.

## Comandos principais na VPS

Entre na pasta do projeto:

```bash
cd /opt/lotofacil
```

Subir ou reconstruir:

```bash
docker compose up -d --build
```

Ver containers:

```bash
docker compose ps
```

Ver logs do calibrador:

```bash
docker compose logs -f --tail=120 lotofacil-calibrator
```

Ver logs da interface:

```bash
docker compose logs -f --tail=120 lotofacil-web
```

Testar status pela VPS:

```bash
curl http://127.0.0.1:8765/api/calibration/status
```

Parar sem apagar dados:

```bash
docker compose down
```

Reiniciar somente o calibrador:

```bash
docker compose restart lotofacil-calibrator
```

## Arquivos importantes

```text
/opt/lotofacil/data/processed/lotofacil_calibration_lab_state.json
/opt/lotofacil/data/processed/lotofacil_calibration_lab_attempts.csv
/opt/lotofacil/data/processed/lotofacil_calibration_lab_winners.csv
/opt/lotofacil/data/processed/lotofacil_calibration_lab_elites.csv
/opt/lotofacil/data/processed/lotofacil_calibration_lab_average_weights.csv
/opt/lotofacil/data/processed/lotofacil_engine_calibrated_weights.json
/opt/lotofacil/data/processed/lotofacil_calibration_lab_cache/
```

## Como reverter

Para parar os containers sem apagar dados:

```bash
cd /opt/lotofacil
docker compose down
```

Para voltar o codigo para o backup criado antes do deploy, use a pasta `/opt/lotofacil_backup_*` gerada no momento da publicacao. Exemplo:

```bash
sudo rm -rf /opt/lotofacil
sudo mv /opt/lotofacil_backup_YYYYMMDD_HHMMSS /opt/lotofacil
cd /opt/lotofacil
docker compose up -d --build
```

Troque `YYYYMMDD_HHMMSS` pelo nome real da pasta de backup.
