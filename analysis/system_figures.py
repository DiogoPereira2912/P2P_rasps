#!/usr/bin/env python3
"""
Figuras de desempenho (artigo 2: GL vs FL) a partir do export de tools/export_metrics.py.
Modo de uso: docs/analysis.md (secção "Métricas de sistema")
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures import paper_style, _save, COL1, COL2, COLORS, SERIES_LABEL
from metrics import mean_ci

MODES = ("gossip", "federated")
# Tipos de mensagem que atravessam a rede entre nós (os outros ficam no broker local)
INTER_NODE = {"agg": "Predições", "train": "Decisões globais", "dataset": "Datasets (merge GT)"}
SERVICES = {"detector_container": "Detetor", "pipeline": "Pipeline", "labeler_container": "Labeler",
            "aggregation": "Consenso", "mqtt_broker": "Broker", "client": "Peer"}


def per_minute(df):
    return df.assign(value=df["value"] / df["duration_s"] * 60)


def modes_in(df):
    return [m for m in MODES if m in df["mode"].unique()]


def bars_by_mode(ax, table, ylabel):
    """table: índice = categoria, colunas = (mode, 'mean'|'ci')."""
    modes = [m for m in MODES if (m, "mean") in table.columns]
    x = np.arange(len(table))
    width = 0.8 / max(1, len(modes))
    for i, mode in enumerate(modes):
        ax.bar(x + (i - (len(modes) - 1) / 2) * width, table[(mode, "mean")], width,
               yerr=table[(mode, "ci")], capsize=2, color=COLORS[mode], label=SERIES_LABEL[mode],
               error_kw={"linewidth": 0.6})
    ax.set_xticks(x, table.index)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)


def summarize(df, by):
    """Média ± IC 95% entre runs, por (categoria, modo)."""
    per_run = df.groupby(["run_id", "mode"] + by)["value"].sum().reset_index()
    rows = []
    for keys, sub in per_run.groupby(by + ["mode"]):
        mean, half = mean_ci(sub["value"])
        rows.append((*keys, mean, half))
    t = pd.DataFrame(rows, columns=by + ["mode", "mean", "ci"])
    return t.pivot_table(index=by[0], columns="mode", values=["mean", "ci"]).swaplevel(axis=1)


# ---------------------------------------------------------------------------
def fig_traffic_by_kind(m, out):
    """S1: mensagens/min entre nós por tipo (total da frota)."""
    d = per_minute(m[(m.metric == "mqtt_msgs_sent") & m.kind.isin(INTER_NODE)])
    t = summarize(d.assign(kind=d.kind.map(INTER_NODE)), ["kind"]).reindex(INTER_NODE.values()).dropna(how="all")
    fig, ax = plt.subplots(figsize=(COL1, 2.0))
    bars_by_mode(ax, t, "Mensagens / min (frota)")
    return _save(fig, out, "figS1_traffic_by_kind")


def fig_traffic_by_node(m, out):
    """S2: mensagens recebidas/min entre nós, por nó (concentração no central em FL)."""
    d = per_minute(m[(m.metric == "mqtt_msgs_received") & m.kind.isin(["agg", "train"])])
    t = summarize(d, ["node"])
    fig, ax = plt.subplots(figsize=(COL1, 2.0))
    bars_by_mode(ax, t, "Mensagens de consenso\nrecebidas / min")
    return _save(fig, out, "figS2_traffic_by_node")


def fig_consensus_cpu(m, out):
    """S3: CPU do serviço de consenso por nó."""
    d = m[(m.metric == "cpu_cores_by_service") & (m.kind == "aggregation")]
    t = summarize(d.assign(value=d.value * 100), ["node"])
    fig, ax = plt.subplots(figsize=(COL1, 2.0))
    bars_by_mode(ax, t, "CPU do consenso (% de 1 core)")
    return _save(fig, out, "figS3_consensus_cpu")


def fig_latency(m, out):
    """S4: latência predição -> decisão e predição -> agregador (p50 / p95)."""
    labels = {"message_delay_p50": "Mensagem p50", "message_delay_p95": "Mensagem p95",
              "decision_delay_p50": "Decisão p50", "decision_delay_p95": "Decisão p95"}
    d = m[m.metric.isin(labels)].assign(metric=lambda x: x.metric.map(labels))
    t = summarize(d, ["metric"]).reindex(labels.values()).dropna(how="all")
    fig, ax = plt.subplots(figsize=(COL1, 2.0))
    bars_by_mode(ax, t, "Latência (s)")
    return _save(fig, out, "figS4_latency")


def fig_resources(m, out):
    """S5: CPU por serviço (média dos nós), GL vs FL — mostra o peso do detetor vs consenso."""
    d = m[(m.metric == "cpu_cores_by_service") & m.kind.isin(SERVICES)]
    d = d.groupby(["run_id", "mode", "kind"])["value"].mean().reset_index()  # média dos nós
    t = d.groupby(["mode", "kind"])["value"].mean().unstack("kind").reindex(columns=list(SERVICES)).fillna(0)
    t = t.reindex([m_ for m_ in MODES if m_ in t.index])
    fig, ax = plt.subplots(figsize=(COL1, 1.9))
    left = np.zeros(len(t))
    palette = ["#999999", "#56B4E9", "#009E73", "#D55E00", "#CC79A7", "#F0E442"]
    for color, service in zip(palette, SERVICES):
        ax.barh([SERIES_LABEL[m_] for m_ in t.index], t[service], left=left, color=color, label=SERVICES[service])
        left += t[service].values
    ax.set_xlabel("CPU médio por nó (cores)")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6)
    return _save(fig, out, "figS5_resources_by_service")


def fig_temperature(m, out):
    """S6: temperatura máxima por nó (viabilidade em edge; throttling do Pi 5 ~85 °C)."""
    d = m[m.metric == "temp_max_c"]
    t = summarize(d, ["node"])
    fig, ax = plt.subplots(figsize=(COL1, 1.9))
    bars_by_mode(ax, t, "Temperatura máx. (°C)")
    ax.axhline(85, color="#D55E00", linestyle="--", linewidth=0.8)
    ax.text(ax.get_xlim()[1], 85.5, "throttling", ha="right", va="bottom", fontsize=6, color="#D55E00")
    lo = max(0, np.nanmin(t.xs("mean", axis=1, level=1).values) - 15)
    ax.set_ylim(lo, 95)
    return _save(fig, out, "figS6_temperature")


def fig_rounds_timeline(ts, out):
    """S7: decisões/min por nó ao longo de cada run (robustez: falha de um nó no cenário B)."""
    paths = []
    d = ts[ts.metric == "rounds_per_min"]
    for run_id, sub in d.groupby("run_id"):
        fig, ax = plt.subplots(figsize=(COL1, 1.8))
        for node, s in sub.groupby("node"):
            ax.plot(s["t_s"], s["value"], linewidth=0.9, label=node)
        ax.set_xlabel("Tempo desde o início do run (s)")
        ax.set_ylabel("Decisões / min")
        ax.legend(frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=6)
        safe = run_id.replace("/", "_")
        paths.append(_save(fig, out, f"figS7_rounds_timeline_{safe}"))
    return paths


def scalability_table(m, out):
    """Se houver runs com nº de nós diferente: mensagens de consenso/min vs nº de nós."""
    n = m[m.metric == "nodes_in_round"].groupby("run_id")["value"].max().round().rename("n_nodes")
    if n.nunique() < 2:
        return None
    msgs = per_minute(m[(m.metric == "mqtt_msgs_sent") & m.kind.isin(["agg", "train"])])
    t = msgs.groupby(["run_id", "mode"])["value"].sum().reset_index().join(n, on="run_id")
    t.to_csv(out / "scalability.csv", index=False)
    fig, ax = plt.subplots(figsize=(COL1, 2.0))
    for mode in modes_in(t):
        s = t[t["mode"] == mode].groupby("n_nodes")["value"].mean()
        ax.plot(s.index, s.values, marker="o", color=COLORS[mode], label=SERIES_LABEL[mode])
    ax.set_xlabel("Nº de nós")
    ax.set_ylabel("Mensagens de consenso / min")
    ax.legend(frameon=False)
    return _save(fig, out, "figS8_scalability")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="pasta do export (experiments/system_metrics/<data>)")
    parser.add_argument("--run-filter", help="só runs cujo run_id contém este texto")
    parser.add_argument("--out", help="pasta das figuras (default: <folder>/figures)")
    args = parser.parse_args()

    folder = Path(args.folder)
    m = pd.read_csv(folder / "system_metrics.csv")
    ts = pd.read_csv(folder / "system_timeseries.csv")
    if args.run_filter:
        m, ts = m[m.run_id.str.contains(args.run_filter)], ts[ts.run_id.str.contains(args.run_filter)]
    m["kind"] = m["kind"].fillna("")
    out = Path(args.out or folder / "figures")
    out.mkdir(parents=True, exist_ok=True)
    paper_style()

    print("Runs:", m.groupby("mode")["run_id"].nunique().to_dict())

    # Tabela resumo (média ± IC 95% entre runs) para o artigo
    rows = []
    for mode in modes_in(m):
        sub = m[m["mode"] == mode]
        def stat(metric, kinds=None, agg="sum", scale=1.0, minute=False):
            d = sub[sub.metric == metric]
            if kinds is not None:
                d = d[d.kind.isin(kinds)]
            if minute:
                d = per_minute(d)
            vals = d.groupby("run_id")["value"].agg(agg) * scale
            mean, half = mean_ci(vals)
            return f"{mean:.3g} ± {half:.2g}"
        rows.append({
            "mode": mode, "runs": sub.run_id.nunique(),
            "consensus_msgs_per_min": stat("mqtt_msgs_sent", ["agg", "train"], minute=True),
            "inter_node_kB_per_min": stat("mqtt_bytes_sent", list(INTER_NODE), minute=True, scale=1 / 1000),
            "decision_delay_p50_s": stat("decision_delay_p50", agg="max"),
            "decision_delay_p95_s": stat("decision_delay_p95", agg="max"),
            "consensus_cpu_max_node_pct": stat("cpu_cores_by_service", ["aggregation"], agg="max", scale=100),
            "temp_max_c": stat("temp_max_c", agg="max"),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "system_summary.csv", index=False)
    print("\n" + summary.to_string(index=False))

    made = [fig_traffic_by_kind(m, out), fig_traffic_by_node(m, out), fig_consensus_cpu(m, out),
            fig_latency(m, out), fig_resources(m, out), fig_temperature(m, out), scalability_table(m, out)]
    made += fig_rounds_timeline(ts, out)
    print("\n📊 Figuras:")
    for p in filter(None, made):
        print(f"   {p}")


if __name__ == "__main__":
    main()
