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

### Métricas (classes desequilibradas: quase tudo é SAFE)

- **Macro-F1 / balanced accuracy / MCC**: não são dominadas pela classe SAFE.
- **Danger recall**: dos instantes com perigo real, quantos foram detetados como perigo.
  É a métrica de segurança.
- **False alarm rate**: instantes SAFE classificados como perigo.
- **McNemar**: diz se a diferença local vs consenso é estatisticamente significativa.

---

## 4. Problemas comuns

| Sintoma | Causa |
|---|---|
| `X/Y predições com ground truth (descartadas: muitas)` | Relógios dessincronizados durante o run, ou run sem fase final de merge do labeler |
| Labeler: `⚠️ Merge vazio` | Relógios dessincronizados → `tools/sync_clocks.py` |
| `mode = desconhecido` | Dados gravados antes desta versão do aggregation |
| Fig. 5 em falta | Dados sem `round_probs` (versão antiga) ou sem `global_rules.yaml` na experiência |
