# Deploy na VPS - Lotofacil Analytics

## Resumo

O projeto roda em dois containers Docker:

1. `lotofacil-web`: interface web na porta `8765`.
2. `lotofacil-calibrator`: calibrador 24/7 que tenta pesos diferentes ate acertar 15 dezenas em um dos dois jogos historicos.

Os arquivos de progresso ficam no volume `./data`, fora da imagem Docker. Se o container reiniciar, o processo continua do estado salvo.

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
/opt/lotofacil/data/processed/lotofacil_calibration_lab_average_weights.csv
/opt/lotofacil/data/processed/lotofacil_engine_calibrated_weights.json
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
