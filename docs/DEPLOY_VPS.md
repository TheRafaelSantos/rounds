# Deploy na VPS - Lotofacil Analytics

## Resumo

O projeto roda em dois containers Docker por padrao:

1. `lotofacil-web`: interface web na porta `8765`.
2. `lotofacil-supervised`: aprendizado supervisionado 24/7 que usa o gabarito de concursos historicos para calibrar a media de pesos aplicada ao motor principal.

Os arquivos de progresso ficam no volume `./data`, fora da imagem Docker. Se o container reiniciar, o processo continua do estado salvo.

O aprendizado supervisionado avalia, para cada concurso historico, a sequencia real contra amostras concorrentes. Ele mede quais estudos deixariam o gabarito mais bem posicionado, calcula pesos por concurso e grava a media em `data/processed/lotofacil_supervised_calibrated_weights.json`. Esse arquivo e o unico arquivo de pesos carregado pelo motor principal ao gerar novos jogos.

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

Ver logs do aprendizado supervisionado:

```bash
docker compose logs -f --tail=120 lotofacil-supervised
```

Ver logs da interface:

```bash
docker compose logs -f --tail=120 lotofacil-web
```

Testar status pela VPS:

```bash
curl http://127.0.0.1:8765/api/supervised/status
```

Parar sem apagar dados:

```bash
docker compose down
```

Reiniciar somente o aprendizado supervisionado:

```bash
docker compose restart lotofacil-supervised
```

## Arquivos importantes

```text
/opt/lotofacil/data/processed/lotofacil_supervised_calibration_state.json
/opt/lotofacil/data/processed/lotofacil_supervised_calibration_results.csv
/opt/lotofacil/data/processed/lotofacil_supervised_calibration_summary.csv
/opt/lotofacil/data/processed/lotofacil_supervised_calibration_weights.csv
/opt/lotofacil/data/processed/lotofacil_supervised_calibrated_weights.json
/opt/lotofacil/data/exports/lotofacil_supervised_calibration.xlsx
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
