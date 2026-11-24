"""
Figuras para o artigo. Convenções:
  - PDF vetorial, fontes TrueType embebidas (pdf.fonttype 42, exigido por IEEE/Elsevier)
  - largura de 1 coluna (3.5") ou 2 colunas (7.16") do template IEEE
  - fonte 8 pt, sem título dentro da figura (o título vai na legenda do artigo)
  - paleta Okabe-Ito (legível por daltónicos)
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metrics import classification_metrics, confusion, mean_ci, SAFE

COL1, COL2 = 3.5, 7.16  # polegadas

# Okabe-Ito
COLORS = {"local": "#999999", "gossip": "#0072B2", "federated": "#E69F00", "extra": "#009E73"}
SERIES_LABEL = {"local": "Local (1 câmara)", "gossip": "Consenso GL", "federated": "Consenso FL"}

# Nomes curtos para os eixos (editar à vontade, ex: para inglês no artigo)
SHORT_NAMES = {
    "SAFE": "SAFE",
    "PERIGO_GRUA": "CRANE",
    "QUEDA_DETETADA": "FALL",
    "SINAL_SOS": "SOS",
    "ZONA_FORKLIFT": "FORKLIFT",
    "PROXIMIDADE_MAQUINA": "MACHINE",
    "INTRUSAO": "INTRUSION",
    # regras v2
    "QUEDA": "FALL",
    "BRACO_ESTENDIDO": "REACH",
}

METRIC_LABELS = {
    "macro_f1": "Macro-F1",
    "balanced_accuracy": "Balanced acc.",
    "mcc": "MCC",
    "danger_recall": "Danger recall",
}


def paper_style():
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def _short(classes):
    return [SHORT_NAMES.get(c, c) for c in classes]


def _save(fig, out_dir, name):
    path = out_dir / f"{name}.pdf"
    fig.savefig(path)
    fig.savefig(out_dir / f"{name}.png")  # pré-visualização rápida
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 1. Local vs consenso (GL / FL), média ± IC 95% entre runs
# ---------------------------------------------------------------------------
def fig_local_vs_consensus(per_run, out_dir, metrics=tuple(METRIC_LABELS), name="fig1_local_vs_consensus"):
    """per_run: DataFrame com 1 linha por (run_id, série) e as métricas em colunas."""
    series = [s for s in ("local", "gossip", "federated") if s in per_run["series"].unique()]
    x = np.arange(len(metrics))
    width = 0.8 / len(series)

    fig, ax = plt.subplots(figsize=(COL1, 2.0))
    for i, s in enumerate(series):
        sub = per_run[per_run["series"] == s]
        stats_ = [mean_ci(sub[m]) for m in metrics]
        ax.bar(x + (i - (len(series) - 1) / 2) * width, [m for m, _ in stats_], width,
               yerr=[h for _, h in stats_], capsize=2, color=COLORS[s],
               label=f"{SERIES_LABEL[s]} (n={sub['run_id'].nunique()})", error_kw={"linewidth": 0.6})
    ax.set_xticks(x, [METRIC_LABELS[m] for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    # legenda por cima do gráfico, para não tapar barras
    ax.legend(frameon=False, ncol=len(series), loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=1, columnspacing=0.8)
    return _save(fig, out_dir, name)


# ---------------------------------------------------------------------------
# 2. Matrizes de confusão normalizadas: local vs consenso
# ---------------------------------------------------------------------------
def fig_confusion(df, classes, mode, out_dir):
    sub = df[df["mode"] == mode]
    if sub.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.9), constrained_layout=True)
    for ax, col, title in zip(axes, ["local_pred", "global_pred"], ["Local", f"Consenso {mode}"]):
        cm = confusion(sub["GLOBAL_STATUS"], sub[col], classes)
        with np.errstate(invalid="ignore"):
            norm = cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(np.nan_to_num(norm), vmin=0, vmax=1, cmap="Blues")
        for i in range(len(classes)):
            for j in range(len(classes)):
                if cm[i, j]:
                    color = "white" if norm[i, j] > 0.6 else "black"
                    ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center",
                            fontsize=5.5, color=color)
        ax.set_xticks(range(len(classes)), _short(classes), rotation=45, ha="right")
        ax.set_yticks(range(len(classes)), _short(classes))
        ax.set_xlabel("Previsto")
        ax.set_title(title)
        ax.grid(False)
    axes[0].set_ylabel("Ground truth")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Proporção por linha")
    return _save(fig, out_dir, f"fig2_confusion_{mode}")


# ---------------------------------------------------------------------------
# 3. Correções vs regressões do consenso, por classe real
# ---------------------------------------------------------------------------
def fig_corrections(df, classes, out_dir):
    modes = [m for m in ("gossip", "federated") if m in df["mode"].unique()]
    if not modes:
        return None
    fig, axes = plt.subplots(1, len(modes), figsize=(COL1 * len(modes) * 0.75, 2.2),
                             sharey=True, squeeze=False)
    for ax, mode in zip(axes[0], modes):
        sub = df[df["mode"] == mode]
        local_ok = sub["local_pred"] == sub["GLOBAL_STATUS"]
        global_ok = sub["global_pred"] == sub["GLOBAL_STATUS"]
        corrected = sub[~local_ok & global_ok]["GLOBAL_STATUS"].value_counts()
        regressed = sub[local_ok & ~global_ok]["GLOBAL_STATUS"].value_counts()
        y = np.arange(len(classes))
        ax.barh(y, [corrected.get(c, 0) for c in classes], color=COLORS["extra"], label="Corrigido pelo consenso")
        ax.barh(y, [-regressed.get(c, 0) for c in classes], color="#D55E00", label="Estragado pelo consenso")
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_yticks(y, _short(classes))
        ax.set_xlabel("Nº de amostras")
        ax.set_title(SERIES_LABEL[mode])
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    return _save(fig, out_dir, "fig3_corrections")


# ---------------------------------------------------------------------------
# 4. Latência predição -> decisão global (ECDF), GL vs FL
# ---------------------------------------------------------------------------
def fig_latency(df, out_dir):
    if not {"pred_ts", "decision_ts"}.issubset(df.columns):
        return None
    lat = df.drop_duplicates(["device", "pred_ts"]).assign(latency=lambda d: d["decision_ts"] - d["pred_ts"])
    fig, ax = plt.subplots(figsize=(COL1, 2.0))
    for mode in ("gossip", "federated"):
        v = np.sort(lat.loc[lat["mode"] == mode, "latency"].dropna().values)
        if len(v):
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=COLORS[mode],
                    label=f"{SERIES_LABEL[mode]} (mediana {np.median(v):.2f} s)")
    ax.set_xlabel("Latência predição → decisão global (s)")
    ax.set_ylabel("ECDF")
    ax.legend(frameon=False, loc="lower right")
    return _save(fig, out_dir, "fig4_latency_ecdf")


# ---------------------------------------------------------------------------
# 5. Comparação de métodos de consenso (recalculados offline, mesmos dados)
# ---------------------------------------------------------------------------
def fig_methods(method_scores, out_dir, metric="macro_f1"):
    """method_scores: DataFrame com colunas method, run_id, <metric>."""
    methods = list(dict.fromkeys(method_scores["method"]))
    stats_ = [mean_ci(method_scores.loc[method_scores["method"] == m, metric]) for m in methods]
    fig, ax = plt.subplots(figsize=(COL1, 1.9))
    colors = [COLORS["local"] if m == "local" else COLORS["gossip"] for m in methods]
    ax.bar(range(len(methods)), [s for s, _ in stats_], yerr=[h for _, h in stats_],
           capsize=2, color=colors, error_kw={"linewidth": 0.6})
    ax.set_xticks(range(len(methods)), methods, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    return _save(fig, out_dir, "fig5_consensus_methods")


def run_metrics(df, classes, pred_col, gt_col="GLOBAL_STATUS"):
    """Métricas por run_id (cada run = uma repetição), contra o ground truth gt_col."""
    rows = []
    for run_id, sub in df.groupby("run_id"):
        m = classification_metrics(sub[gt_col], sub[pred_col], classes)
        m["run_id"] = run_id
        m["mode"] = sub["mode"].iloc[0]
        rows.append(m)
    return pd.DataFrame(rows)
