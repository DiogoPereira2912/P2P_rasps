#!/usr/bin/env python3
"""
Exporta do Prometheus as métricas de sistema de cada run (artigo de desempenho GL vs FL).
Os runs são detetados pelo indicador aidi_run_active (do INFERENCE ao STOP, com o run_id).
Modo de uso: docs/analysis.md (secção "Métricas de sistema")
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
# No Mac: localhost. No contentor da UI: http://prometheus:9090 (variável de ambiente)
PROM = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
INTER_NODE_KINDS = ["agg", "train", "dataset"]  # mensagens que atravessam a rede entre nós
MAX_POINTS = 10000  # limite de pontos por query_range do Prometheus (11000), com margem


def prom(path, **params):
    url = f"{PROM}/api/v1/{path}?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.load(r)
    if data["status"] != "success":
        raise RuntimeError(data)
    return data["data"]["result"]


def parse_duration(text):
    units = {"m": 60, "h": 3600, "d": 86400}
    return float(text[:-1]) * units[text[-1]]


def find_runs(since_s, run_filter=None):
    """Intervalos [início, fim] de cada run, a partir de aidi_run_active == 1."""
    end = time.time()
    start = end - since_s
    step = max(5, int(since_s / MAX_POINTS))
    series = prom("query_range", query="max by (run_id, mode) (aidi_run_active)",
                  start=start, end=end, step=step)
    runs = []
    for s in series:
        run_id, mode = s["metric"]["run_id"], s["metric"]["mode"]
        if run_filter and run_filter not in run_id:
            continue
        active = [float(t) for t, v in s["values"] if float(v) == 1]
        if active:
            runs.append({"run_id": run_id, "mode": mode,
                         "start": min(active), "end": max(active) + step})
    return sorted(runs, key=lambda r: r["start"])


def instant(query, at):
    """Query instantânea num instante (fim do run). Devolve [(labels, valor)]."""
    return [(r["metric"], float(r["value"][1])) for r in prom("query", query=query, time=at)]


def run_metrics(run):
    """Métricas agregadas de UM run (tudo calculado dentro da janela do run)."""
    at, dur = run["end"], run["end"] - run["start"]
    w = f"{int(dur)}s"
    mode = run["mode"]
    rows = []

    def add(metric, query, label="node", kind_label=None, scale=1.0):
        for labels, value in instant(query, at):
            rows.append({"metric": metric, "node": labels.get(label, "all"),
                         "kind": labels.get(kind_label, "") if kind_label else "",
                         "value": value * scale})

    # --- Comunicação (MQTT): totais do run e por minuto ---
    for direction in ("sent", "received"):
        add(f"mqtt_bytes_{direction}",
            f'sum by (node, kind) (increase(aidi_mqtt_bytes_total{{direction="{direction}"}}[{w}]))',
            kind_label="kind")
        add(f"mqtt_msgs_{direction}",
            f'sum by (node, kind) (increase(aidi_mqtt_messages_total{{direction="{direction}"}}[{w}]))',
            kind_label="kind")
    # --- Consenso ---
    # Filtrado pelo modo do run: evita contar rondas de um run anterior noutro modo
    # que caiam no limite da janela
    add("rounds", f'sum by (node) (increase(aidi_agg_rounds_total{{mode="{mode}"}}[{w}]))')
    add("divergence", f'sum by (node) (increase(aidi_agg_divergence_total{{mode="{mode}"}}[{w}]))')
    add("nodes_in_round", f"avg by (node) (avg_over_time(aidi_agg_nodes_in_round[{w}]))")
    for q in (0.5, 0.95):
        add(f"decision_delay_p{int(q * 100)}",
            f'histogram_quantile({q}, sum by (le) (increase(aidi_agg_decision_delay_seconds_bucket{{mode="{mode}"}}[{w}])))',
            label="__none__")
        add(f"message_delay_p{int(q * 100)}",
            f'histogram_quantile({q}, sum by (le) (increase(aidi_agg_message_delay_seconds_bucket{{mode="{mode}"}}[{w}])))',
            label="__none__")
    # --- Recursos ---
    add("cpu_cores_by_service",
        f'sum by (node, name) (increase(container_cpu_usage_seconds_total{{name!=""}}[{w}])) / {dur}',
        kind_label="name")
    add("cpu_pct_node",
        f'100 * sum by (node) (increase(container_cpu_usage_seconds_total{{id="/"}}[{w}])) / {dur} '
        f'/ sum by (node) (machine_cpu_cores)')
    add("ram_bytes_by_service",
        f'avg by (node, name) (avg_over_time(container_memory_working_set_bytes{{name!=""}}[{w}]))',
        kind_label="name")
    add("ram_bytes_node", f'avg by (node) (avg_over_time(container_memory_working_set_bytes{{id="/"}}[{w}]))')
    add("temp_mean_c", f"avg by (node) (avg_over_time(aidi_cpu_temp_celsius[{w}]))")
    add("temp_max_c", f"max by (node) (max_over_time(aidi_cpu_temp_celsius[{w}]))")

    df = pd.DataFrame(rows)
    for k in ("run_id", "mode"):
        df.insert(0, k, run[k])
    df["duration_s"] = dur
    return df


def kpis(run_df, ts_df=None):
    """
    Números principais de UM run (para a UI e para comparar GL vs FL de relance).
    run_df: as linhas desse run devolvidas por run_metrics(); ts_df: run_timeseries() (opcional).
    Inclui o CUSTO (tráfego) e os BENEFÍCIOS (equilíbrio, nós que decidem, disponibilidade).
    """
    d, dur = run_df, run_df["duration_s"].iloc[0]

    def total(metric, kinds=None):
        x = d[d.metric == metric]
        if kinds is not None:
            x = x[x.kind.isin(kinds)]
        return x["value"].sum()

    def one(metric):
        x = d[d.metric == metric]["value"]
        return float(x.iloc[0]) if len(x) else float("nan")

    cpu = d[(d.metric == "cpu_cores_by_service") & (d.kind == "aggregation")].set_index("node")["value"] * 100
    temp = d[d.metric == "temp_max_c"]["value"]

    # Carga de consenso recebida por nó (msgs/min): equilíbrio entre nós
    recv = (d[(d.metric == "mqtt_msgs_received") & d.kind.isin(["agg", "train"])]
            .groupby("node")["value"].sum() / dur * 60)
    rounds = d[d.metric == "rounds"].set_index("node")["value"]
    deciders = int((rounds >= 1).sum())  # nós com pelo menos 1 decisão neste run
    n_nodes = int(round(d[d.metric == "nodes_in_round"]["value"].max())) if (d.metric == "nodes_in_round").any() else None
    divergence = d[d.metric == "divergence"]["value"].sum()

    # Disponibilidade: % do run com pelo menos uma decisão global a ser tomada na frota
    # (ignora os primeiros 30 s, janela da taxa no Prometheus)
    availability = None
    if ts_df is not None and not ts_df.empty:
        r = ts_df[(ts_df.metric == "rounds_per_min") & (ts_df.t_s >= 30)].groupby("t_s")["value"].sum()
        if len(r):
            availability = round(100 * (r > 0).mean(), 1)

    return {
        "run_id": d["run_id"].iloc[0],
        "modo": d["mode"].iloc[0],
        "duração (s)": round(dur),
        "nós": int(round(d[d.metric == "nodes_in_round"]["value"].max())) if (d.metric == "nodes_in_round").any() else None,
        "msgs consenso/min": round(total("mqtt_msgs_sent", ["agg", "train"]) / dur * 60),
        "kB entre nós/min": round(total("mqtt_bytes_sent", INTER_NODE_KINDS) / dur * 60 / 1000, 1),
        "latência decisão p50 (s)": round(one("decision_delay_p50"), 2),
        "latência decisão p95 (s)": round(one("decision_delay_p95"), 2),
        "CPU consenso máx nó (%)": round(cpu.max(), 2) if len(cpu) else None,
        "CPU consenso mín nó (%)": round(cpu.min(), 2) if len(cpu) else None,
        "temp. máx (°C)": round(temp.max(), 1) if len(temp) else None,
        # --- benefícios / robustez ---
        "nós que decidem": f"{deciders}/{n_nodes}" if n_nodes else deciders,
        "ponto único de falha": "sim" if deciders <= 1 else "não",
        "carga nó mais carregado (msgs/min)": round(recv.max()) if len(recv) else None,
        "desequilíbrio de carga (máx/mín)": round(recv.max() / recv.min(), 1) if len(recv) and recv.min() > 0 else None,
        "disponibilidade decisão (%)": availability,
        "consenso alterou local (%)": round(100 * divergence / rounds.sum(), 1) if rounds.sum() > 0 else None,
    }


def run_timeseries(run, step=5):
    """Séries temporais dentro do run (ex: robustez: rondas por nó ao longo do tempo)."""
    queries = {
        "rounds_per_min": "60 * sum by (node) (rate(aidi_agg_rounds_total[30s]))",
        "mqtt_bytes_sent_per_s": 'sum by (node) (rate(aidi_mqtt_bytes_total{direction="sent"}[30s]))',
        "cpu_pct_node": '100 * sum by (node) (rate(container_cpu_usage_seconds_total{id="/"}[30s])) '
                        '/ sum by (node) (machine_cpu_cores)',
        "temp_c": "aidi_cpu_temp_celsius",
    }
    rows = []
    for name, q in queries.items():
        for s in prom("query_range", query=q, start=run["start"], end=run["end"], step=step):
            node = s["metric"].get("node", "all")
            for t, v in s["values"]:
                rows.append({"run_id": run["run_id"], "mode": run["mode"], "metric": name, "node": node,
                             "t_s": float(t) - run["start"], "value": float(v)})
    return pd.DataFrame(rows)


def main():
    global PROM
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default="24h", help="procurar runs nas últimas ... (ex: 6h, 2d)")
    parser.add_argument("--run-filter", help="só runs cujo run_id contém este texto")
    parser.add_argument("--run-ids", help="só estes runs (run_id separados por vírgula)")
    parser.add_argument("--out", help="pasta de saída (default: experiments/system_metrics/<data>)")
    parser.add_argument("--prometheus", default=PROM)
    args = parser.parse_args()
    PROM = args.prometheus

    runs = find_runs(parse_duration(args.since), args.run_filter)
    if args.run_ids:
        wanted = set(args.run_ids.split(","))
        runs = [r for r in runs if r["run_id"] in wanted]
    if not runs:
        sys.exit(f"❌ Nenhum run encontrado nas últimas {args.since} (o aggregation novo está nos Pis?)")

    out = Path(args.out or ROOT / "experiments" / "system_metrics" / datetime.now().strftime("%Y-%m-%d_%H%M"))
    out.mkdir(parents=True, exist_ok=True)

    print(f"{'run_id':45s} {'modo':10s} {'duração':>8s}")
    metrics, series = [], []
    for run in runs:
        print(f"{run['run_id']:45s} {run['mode']:10s} {run['end'] - run['start']:7.0f}s")
        metrics.append(run_metrics(run))
        series.append(run_timeseries(run))

    pd.DataFrame(runs).assign(
        start=lambda d: pd.to_datetime(d["start"], unit="s"), end=lambda d: pd.to_datetime(d["end"], unit="s")
    ).to_csv(out / "runs.csv", index=False)
    pd.concat(metrics, ignore_index=True).to_csv(out / "system_metrics.csv", index=False)
    pd.concat(series, ignore_index=True).to_csv(out / "system_timeseries.csv", index=False)
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"\n📁 {shown}/  (runs.csv, system_metrics.csv, system_timeseries.csv)")
    print(f"   Figuras: python3 analysis/system_figures.py {shown}")


if __name__ == "__main__":
    main()
