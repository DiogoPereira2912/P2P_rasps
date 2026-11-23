# P2P Gossip Learning — Raspberry Pi 5

Arquitetura de **Gossip Learning** (com modo **Federated Learning** para comparação) a correr numa frota de Raspberry Pi 5 com acelerador Hailo. Cada Pi deteta pessoas/poses pela câmara, gera labels por regras, treina um modelo local e troca predições com os outros nós via MQTT, sem servidor central (no modo gossip).

---

## Estrutura do repositório

```
P2P_rasps/
├── README.md                ← este ficheiro
├── docs/                    ← guias de utilização (ver tabela em baixo)
├── inventory.yaml           ← lista de Raspberries (IP, node_id, nome) — fonte única
├── inventory.secrets.yaml   ← tokens (local, NÃO vai para o git)
├── tools/                   ← gen_configs, deploy, sync_clocks, fetch_results (correm no Mac)
├── analysis/                ← figuras e tabelas do artigo (correm no Mac)
│
│   ── Código que vai para os Raspberries (/home/aidi/P2P) ──
├── compose.yaml             ← todos os serviços de um nó
├── client/                  ← peer: descoberta de nós (UDP) + bridges Mosquitto
├── code_detector/           ← deteção YOLO pose
├── labeling/                ← labels por regras (rules.yaml / global_rules.yaml)
├── pipeline_layer/          ← treino e inferência (RandomForest)
├── aggregation_layer/       ← agregação das predições dos nós
├── video_server/            ← stream das câmaras + receção das ROIs
├── mosquitto/               ← config do broker MQTT
│
│   ── Só no Mac (nunca vai para os Pis) ──
├── control/                 ← UI Streamlit + Prometheus + Grafana + scripts de imagens
└── roi_selection/           ← seleção das ROIs das câmaras
```

O que vai e o que não vai para os Pis é definido no `.rsyncignore`. Os dados (`data_exports/`) e os modelos (`models/`) existem **só nos Pis** e não vão para o git.

---

## Guias

| Guia | Para quê |
|---|---|
| [docs/deploy.md](docs/deploy.md) | Inventário, gerar configs, enviar código e imagens para os Pis |
| [docs/control.md](docs/control.md) | UI Streamlit (fases da experiência), Prometheus e Grafana |
| [docs/rois.md](docs/rois.md) | Selecionar e enviar as ROIs de cada câmara |
| [docs/analysis.md](docs/analysis.md) | Protocolo experimental (GL vs FL), métricas e figuras do artigo |
| [docs/cenario.md](docs/cenario.md) | Guião das experiências: calibração das regras, cenários, repetições |

---

## Fluxo de uma experiência (resumo)

1. **Ligar os Pis** e confirmar os IPs no `inventory.yaml`.
2. **Levantar a UI no Mac**: `cd control && docker compose up -d --build` → http://localhost:8501
3. **Enviar o código e a config**: UI → tab *Setup & Deploy* → 📤 Atualizar Código (ou `tools/deploy.sh`) → [deploy.md](docs/deploy.md)
4. **Levantar os serviços nos Pis**: UI → tab *Setup & Deploy* → 🔌 Levantar Serviço
5. **Definir as ROIs** de cada Pi → [rois.md](docs/rois.md)
6. **Sincronizar os relógios**: `python3 tools/sync_clocks.py` → [deploy.md](docs/deploy.md#7-sincronizar-relógios--toolssync_clockspy)
7. Na UI, por ordem: **Data Collection → Treino → Inferência (GL ou FL) → Metrics**
8. **Trazer os resultados para o Mac**: `tools/fetch_results.sh <nome_experiência>`
9. **Figuras e tabelas**: `python3 analysis/make_figures.py experiments/<pasta>` → [analysis.md](docs/analysis.md)

---

## Arquitetura de um nó

```
câmara → detector ──internal/raw_detections──▶ labeler
                                                  │ system/inference   (dados para classificar)
                                                  │ <nó>/dataset       (troca de datasets na recolha)
                                                  ▼
peer ──system/peers──▶ todos           pipeline ──<nó>/agg──▶ aggregation ──<nó>/train──▶
```

| Tópico MQTT | Quem publica | Quem ouve | Conteúdo |
|---|---|---|---|
| `system/control/` | UI Streamlit | todos os serviços | `{"command": COLLECTION/TRAIN/INFERENCE/METRICS/STOP, "config": {...}}` |
| `system/peers` | peer | pipeline, aggregation, labeler | lista de nós descobertos `[ip, node_id, device_id]` |
| `internal/raw_detections` | detector | labeler | deteções brutas |
| `system/inference` | labeler | pipeline | features para classificar |
| `<nó>/dataset` | labeler | labeler (outros nós) | dados labelled para o merge global |
| `<nó>/agg` | pipeline | aggregation | predições + probabilidades locais |
| `<nó>/train` | aggregation | pipeline | predição agregada final |

`<nó>` = IP com `_` em vez de `.` (ex: `192_168_0_110`). Os nós falam entre si através de bridges Mosquitto criadas automaticamente pelo `peer`.

**Modos** (escolhidos na UI, tab Inferência):
- **Gossip (GL)** — cada nó envia para os nós da sua `routing_topology` (por omissão, todos os outros).
- **Federated (FL)** — todos enviam para o servidor central escolhido na UI, que agrega e redistribui.

---

## CHECKPOINT

-> Infraestrutura testada com nova imagem base
-> Configs por nó geradas a partir do `inventory.yaml`; UI, Prometheus e ROIs leem o mesmo inventário

## TODO
-> Testar o deploy offline
-> Testar `tools/deploy.sh`, botões da UI (`node.sh`) e `images.sh` com os Pis ligados
-> Testar integração com o prometheus e grafana
-> Tentar tornar isto o mais profissional e organizado possivel
-> Alterar camada para integração com DL models e aprendizagem automatica com troca de pesos e model pruning
