# Análise e figuras para o artigo

Os gráficos do artigo são feitos **no Mac**, com os dados de todos os nós juntos
(pasta `analysis/`). A fase METRICS nos Pis fica só como verificação rápida.

```
Pis (data_exports/)  ──tools/fetch_results.sh──▶  experiments/<data>_<nome>/  ──analysis/make_figures.py──▶  figures/*.pdf + *.csv
```

---

## 1. O que é comparado

Cada amostra (um instante de uma câmara) tem 3 valores:

| Coluna | O que é | Quem calcula |
|---|---|---|
| `local_pred` | o que **um nó sozinho** conclui com o seu modelo | pipeline do nó |
| `global_pred` | o que os nós concluem **em conjunto** (GL ou FL) | aggregation |
| `GLOBAL_STATUS` | **ground truth** do shop floor (todas as câmaras) | labeler |

O ground truth e o consenso usam **o mesmo critério** (`labeling/global_rules.example.yaml`):
uma regra dispara se ≥ `min_nos` câmaras a virem; entre as que disparam vence a de maior
prioridade; nenhuma → SAFE. A diferença é a fonte: o GT usa as **regras** sobre as deteções,
o consenso usa as **probabilidades dos modelos**.

O modelo local é treinado com o label **local** (o que a sua câmara vê). A hipótese do artigo:
**`global_pred` aproxima-se mais do ground truth do shop floor do que `local_pred`.**

---

## 2. Protocolo para uma comparação GL vs FL válida

1. **Fixar tudo o resto.** Entre runs a comparar, não mudar `global_rules`, `rules.yaml`, ROIs,
   modelos nem nº de nós. Mudar **uma** variável de cada vez (GL ↔ FL, nº de nós, ...).
2. **Mesmo cenário.** Usar um guião de cenário (ex: "cenário A: 0–30 s nada, 30–45 s queda na
   zona do rasp3, ...") e repeti-lo igual em GL e em FL.
3. **Repetições.** Pelo menos 3–5 runs por configuração, alternando (GL, FL, GL, FL, ...) para
   não confundir o efeito do modo com o cansaço/ordem do cenário.
4. **Etiquetar cada run na UI** (tab Inferência → "Etiqueta da experiência"), ex: `cenarioA_rep1`.
   O `run_id` fica `<data_hora>_<GL|FL>_<etiqueta>` e vai em todas as linhas gravadas.
5. **Antes de cada sessão:** `python3 tools/sync_clocks.py` (todos ✅).
6. **Depois de cada sessão:** `tools/fetch_results.sh <nome>` (guarda também as regras e o commit).

### Métodos de consenso: não é preciso repetir experiências

Cada linha gravada guarda as probabilidades de **todos** os nós da ronda (`round_probs`).
A análise recalcula offline vários métodos sobre **exatamente os mesmos dados**
(média, `min_nos` = 1/2/3, ...) → figura 5. Comparar métodos é por isso sempre válido,
mesmo com uma só experiência.

### O que esperar de GL vs FL

Com a topologia completa, GL e FL recebem as mesmas predições e decidem o mesmo: diferenças
de precisão entre runs GL e FL vêm sobretudo do cenário (por isso as repetições). As
diferenças **estruturais** estão na latência (fig. 4), no tráfego (Grafana) e na robustez:
desligar o nó central a meio de um run FL vs desligar um nó qualquer em GL.

---

## 3. Gerar as figuras

```bash
# Uma experiência
python3 analysis/make_figures.py experiments/2026-11-24_1030_cenarioA

# Várias sessões juntas (ex: repetições em dias diferentes)
python3 analysis/make_figures.py experiments/*_cenarioA --out figures/cenarioA

# Só alguns runs (pelo texto do run_id)
python3 analysis/make_figures.py experiments/... --run-filter cenarioA
```

Requer no Mac: `pip install numpy pandas pyarrow matplotlib scipy pyyaml` (não precisa de
deltalake nem sklearn).

### Saída (`<experiência>/figures/`)

| Ficheiro | Conteúdo |
|---|---|
| `fig1_local_vs_consensus.pdf` | Macro-F1, balanced acc., MCC, danger recall: local vs GL vs FL (média ± IC 95% entre runs) |
| `fig2_confusion_<modo>.pdf` | Matrizes de confusão normalizadas, local vs consenso |
| `fig3_corrections.pdf` | Por classe: amostras corrigidas vs estragadas pelo consenso |
| `fig4_latency_ecdf.pdf` | Latência predição → decisão global, GL vs FL |
| `fig5_consensus_methods.pdf` | Métodos de consenso recalculados offline (mesmos dados) |
| `metrics_summary.csv` | Tabela para o artigo (média ± IC 95%) |
| `metrics_per_run.csv` | Métricas de cada run |
| `mcnemar_local_vs_consensus.csv` | Teste estatístico local vs consenso (mesmas amostras) |
| `consensus_methods.csv` | Métricas por método de consenso |

Cada figura sai em **PDF** (vetorial, para o LaTeX) e **PNG** (pré-visualização).
Tamanho de coluna IEEE (3.5" / 7.16"), fonte 8 pt, sem título (vai na legenda do artigo).
Os nomes curtos das classes (ex: `QUEDA_DETETADA` → `FALL`) estão em `SHORT_NAMES` em
`analysis/figures.py`.

### Ground truth humano

Se existirem registos do `tools/event_logger.py` (por defeito em `experiments/event_logs/`, ou
`--events <ficheiros>`), cada predição recebe também o `HUMAN_GT`: o estado esperado do shop
floor no segmento do guião em que caiu (coluna `esperado_min_nos_2` do guião). Saídas extra:

| Ficheiro | Conteúdo |
|---|---|
| `fig1b_local_vs_consensus_humanGT.pdf` | Igual à fig. 1, mas contra o que realmente aconteceu |
| `metrics_summary_humanGT.csv`, `metrics_per_run_humanGT.csv` | Métricas contra o GT humano |
| `rules_vs_humanGT.csv` | Qualidade das próprias regras: GT das regras vs GT humano |

A ligação registo ↔ run é pela etiqueta: o `run_id` termina em `_<tag>`.

### Métricas (classes desequilibradas: quase tudo é SAFE)

- **Macro-F1 / balanced accuracy / MCC**: não são dominadas pela classe SAFE.
- **Danger recall**: dos instantes com perigo real, quantos foram detetados como perigo.
  É a métrica de segurança.
- **False alarm rate**: instantes SAFE classificados como perigo.
- **McNemar**: diz se a diferença local vs consenso é estatisticamente significativa.

---

## 3b. Métricas de sistema (artigo 2: GL vs FL)

As métricas de desempenho vêm do Prometheus (não dos Pis). Cada run é detetado pelo
`aidi_run_active` (do INFERENCE ao STOP) e as métricas são calculadas **dentro da janela do run**.

```bash
python3 tools/export_metrics.py --since 24h                      # todos os runs das últimas 24 h
python3 tools/export_metrics.py --since 2d --run-filter cenarioA # só alguns
python3 analysis/system_figures.py experiments/system_metrics/<data>
```

O Prometheus guarda 180 dias, por isso o export pode ser feito mais tarde (com o Mac ligado ao
Prometheus local, os Pis não precisam de estar ligados).

| Ficheiro | Conteúdo |
|---|---|
| `runs.csv` | run_id, modo, início, fim |
| `system_metrics.csv` | por run e nó: mensagens/bytes por tipo, CPU/RAM por serviço, latências p50/p95, rondas, temperatura |
| `system_timeseries.csv` | séries dentro de cada run (decisões/min, tráfego, CPU, temperatura por nó) |
| `figures/system_summary.csv` | tabela para o artigo (média ± IC 95% entre runs) |
| `figS1_traffic_by_kind` | mensagens/min entre nós por tipo (predições, decisões, datasets) |
| `figS2_traffic_by_node` | mensagens de consenso recebidas por nó (concentração no central em FL) |
| `figS3_consensus_cpu` | CPU do consenso por nó |
| `figS4_latency` | latência mensagem e decisão (p50/p95) |
| `figS5_resources_by_service` | CPU por serviço: peso do detetor vs consenso |
| `figS6_temperature` | temperatura máxima por nó (linha de throttling a 85 °C) |
| `figS7_rounds_timeline_<run>` | decisões/min por nó ao longo do run (robustez, cenário B) |
| `figS8_scalability` | mensagens de consenso vs nº de nós (só se houver runs com nº de nós diferente) |

Tráfego **entre nós** = `agg` (predições), `train` (decisões), `dataset` (merge do GT).
`inference`, `raw_detections` e `peers` ficam no broker local de cada Pi.

---

## 4. Problemas comuns

| Sintoma | Causa |
|---|---|
| `X/Y predições com ground truth (descartadas: muitas)` | Relógios dessincronizados durante o run, ou run sem fase final de merge do labeler |
| Labeler: `⚠️ Merge vazio` | Relógios dessincronizados → `tools/sync_clocks.py` |
| `mode = desconhecido` | Dados gravados antes desta versão do aggregation |
| Fig. 5 em falta | Dados sem `round_probs` (versão antiga) ou sem `global_rules.yaml` na experiência |
