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
| 5. Metrics | 🚀 Metrics Generation | comando `METRICS` — gera os resultados de desempenho |
| todas | ⏹ Parar | comando `STOP` |

Os comandos são publicados em `system/control/` no broker de **cada** IP da lista.

---

## Prometheus e Grafana

O `control/prometheus.yaml` é **gerado** pelo `tools/gen_configs.py` a partir do inventário — não editar à mão. Recolhe de cada nó:

| Job | Porta no Pi | Origem |
|---|---|---|
| `hardware_cadvisor` | 8080 | cAdvisor (CPU, RAM, rede por contentor) |
| `throughput_python` | 8000 | pipeline (`aidi_frames_processed_total`) |

Depois de mudar o inventário:
```bash
python3 tools/gen_configs.py
cd control && docker compose restart prometheus
```

---

## Imagens `.tar`

Ficam em `control/imgs/` — não vão para o git nem para os Pis por rsync. Criar/atualizar com `control/images.sh` (ver [deploy.md](deploy.md#4-imagens-docker-deploy-offline)).
