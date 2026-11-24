# Control Tower — UI Streamlit, Prometheus e Grafana

Corre **só no Mac** (pasta `control/`). Comunica com os Pis por MQTT (comandos) e SSH (deploy).

## Arrancar

```bash
cd control
docker compose up -d --build
```

| Serviço | Endereço | Notas |
|---|---|---|
| UI Streamlit | http://localhost:8501 | controlo da experiência |
| Prometheus | http://localhost:9090 | métricas dos nós |
| Grafana | http://localhost:3000 | dashboards (login `admin` / `admin`) |

O contentor da UI monta:
- `control/` em `/app` (scripts e `imgs/`),
- a raiz do repo em `/repo` (`inventory.yaml` + `tools/deploy.sh`; com escrita, porque o deploy regenera `build/` e `control/prometheus.yaml`),
- `~/.ssh` (só leitura) para conseguir fazer SSH aos Pis.

Para correr a UI fora do Docker: `cd control/streamlit && streamlit run gossip_ui.py`.

Depois de alterar `control/streamlit/requirements.txt`: `docker compose up -d --build control_center`.

---

## A UI

### Barra lateral
- **Raspberry IPs** — vem do `inventory.yaml`. Pode ser editada para testes pontuais (ex: só 2 nós), sem mexer no inventário.
- **Porta MQTT** — vem do inventário (`mosquitto_port`).
- **Parâmetros de recolha / inferência** — intervalo, nº de batches e inferências por batch, enviados com os comandos.

### Tabs (ordem normal de uma experiência)

| Tab | Botão | O que acontece |
|---|---|---|
| 1. Setup & Deploy | 🔌 Levantar Serviço | `node.sh start` em cada nó selecionado (`docker compose down && up -d`) |
| | 📤 Atualizar Código | `tools/deploy.sh <nó>` em cada nó selecionado (+ `--restart` se a opção estiver ativa). Só aceita nós que estão no `inventory.yaml` |
| | 🚀 Deploy Images | `node.sh load` — envia e carrega o `.tar` escolhido de `control/imgs/` |
| 2. Data Collection | 🔴 Iniciar Gravação | comando `COLLECTION` — os nós gravam dados raw e trocam datasets |
| 3. Treino | 🏋️ Iniciar Treino | comando `TRAIN` — cada nó treina o seu modelo com `local_output_<nó>` e guarda em `models/` |
| 4. Inferência | 🚀 Iniciar Inferência | comando `INFERENCE` com o modo **GL** ou **FL** (+ servidor central em FL) |
| 5. Metrics | 🚀 Metrics Generation | comando `METRICS` — relatórios rápidos em cada Pi (`results/`) |
| | 📊 Resultados de sistema | escolher runs → tabela por run + comparação GL vs FL (mensagens, latência, CPU do consenso), complementada com robustez e distribuição (nós que decidem, ponto único de falha, desequilíbrio de carga, correções do consenso). Do Prometheus; Pis podem estar desligados |
| | 📥 Exportar métricas + figuras de sistema | `tools/export_metrics.py` + `analysis/system_figures.py` → `experiments/system_metrics/<data>/` no Mac, figuras mostradas na página |
| | 📥 Trazer dados dos Pis + figuras de classificação | `tools/fetch_results.sh` + `analysis/make_figures.py` → `experiments/<data>_<nome>/` no Mac (Pis ligados) |
| todas | ⏹ Parar | comando `STOP` |

Os comandos são publicados em `system/control/` no broker de **cada** IP da lista.

---

## Prometheus e Grafana

O `control/prometheus.yaml` é **gerado** pelo `tools/gen_configs.py` a partir do inventário — não editar à mão.
Cada série tem a etiqueta `node` (`rasp1`, ...). Recolhe de cada nó:

| Job | Porta no Pi | Origem |
|---|---|---|
| `hardware_cadvisor` | 8080 | cAdvisor (CPU, RAM, rede por contentor) |
| `throughput_python` | 8000 | pipeline (`aidi_frames_processed_total`) |
| `aggregation` | 8002 | consenso: rondas, decisões por classe, divergência local≠global, latências, run ativo |
| `labeler` | 8003 | tráfego MQTT do labeler (troca de datasets) |

Além disso, pipeline, aggregation e labeler expõem o **tráfego MQTT** (`aidi_mqtt_messages_total`,
`aidi_mqtt_bytes_total`, por serviço, direção e tipo de mensagem) e o pipeline a **temperatura do CPU**
(`aidi_cpu_temp_celsius`).

Depois de mudar o inventário (sem reiniciar o Prometheus):
```bash
python3 tools/gen_configs.py
curl -X POST localhost:9090/-/reload
```

As métricas ficam guardadas 180 dias (volume `control_prometheus_data`).

### Dashboards (pasta "Gossip" no Grafana)

| Dashboard | Conteúdo |
|---|---|
| **Gossip · Performance do Sistema** | nós online, CPU/RAM por nó e por serviço, tráfego de rede, frames/s, latência MQTT, temperatura, mensagens e bytes por tipo, tráfego total por run |
| **Gossip · Classificação & Consenso** | rondas/min, decisões globais por classe, divergência local≠global, latência de decisão, GL vs FL |

Ambas têm filtros por nó (e por modo, na de classificação) e mostram cada **run como uma faixa azul**
(do INFERENCE ao STOP, com o `run_id` ao passar o rato). Para ver as métricas de um run, selecionar
a faixa com o rato no gráfico (zoom ao intervalo). A precisão contra o ground truth
não é em tempo real: está nas figuras do [analysis.md](analysis.md).

**As dashboards são ficheiros** (`control/grafana/dashboards/*.json`), carregados automaticamente
no arranque juntamente com a datasource (`control/grafana/provisioning/`). Por isso sobrevivem
a resets do Grafana, mudanças de password e de volumes, e ficam no git.

- Depois de editar os JSON: `cd control && docker compose restart grafana` (o Grafana só relê no arranque).
- Alterar uma dashboard na UI funciona, mas **não altera o ficheiro**. Para guardar:
  *Share → Export → Save to file* e substituir o JSON em `control/grafana/dashboards/`.
- **Mudar a password sem perder nada:**
  ```bash
  docker exec -it grafana grafana cli admin reset-admin-password <nova_password>
  ```
  (O `GF_SECURITY_ADMIN_PASSWORD` do compose só conta na primeira criação do volume.)
- Os volumes têm nome fixo (`control_grafana_data`, `control_prometheus_data`). Antes não tinham,
  e o Docker dava-lhes o nome da pasta: ao mudar a UI de `~/AIDI/Gossip_UI` para `control/`
  o Grafana arrancou com um volume novo e vazio. Os volumes antigos `gossip_ui_*` já não são
  usados (podem ser apagados com `docker volume rm gossip_ui_grafana_data gossip_ui_prometheus_data`).

---

## Imagens `.tar`

Ficam em `control/imgs/` — não vão para o git nem para os Pis por rsync. Criar/atualizar com `control/images.sh` (ver [deploy.md](deploy.md#4-imagens-docker-deploy-offline)).
