#!/usr/bin/env python3
"""
Gera as figuras e tabelas do artigo a partir de uma ou mais pastas experiments/...
Modo de uso: docs/analysis.md
"""
import argparse
import copy
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                        # labeling.labeling_utils
sys.path.insert(0, str(ROOT / "aggregation_layer"))  # aggregation_algs (mesmo código dos Pis)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregation_algs.algs import aggregate_avg, rules_aggregation
from experiment import load_experiments, recompute_consensus
from figures import (paper_style, fig_local_vs_consensus, fig_confusion, fig_corrections,
                     fig_latency, fig_methods, run_metrics)
from metrics import classification_metrics, mcnemar, mean_ci

SERIES_METRICS = ["macro_f1", "balanced_accuracy", "mcc", "danger_recall", "false_alarm_rate", "accuracy"]


def class_list(rules, df):
    if rules:
        return ["SAFE"] + [r["label"] for r in sorted(rules["regras"], key=lambda r: r["prioridade"])]
    return sorted(set(df["GLOBAL_STATUS"]) | set(df["local_pred"]) | set(df["global_pred"]))


def per_run_series(df, classes):
    """1 linha por (run, série): série = local | gossip | federated."""
    local = run_metrics(df, classes, "local_pred").assign(series="local")
    consensus = run_metrics(df, classes, "global_pred")
    consensus["series"] = consensus["mode"]
    return pd.concat([local, consensus], ignore_index=True)


def summary_table(per_run):
    rows = []
    for series, sub in per_run.groupby("series", sort=False):
        row = {"series": series, "runs": sub["run_id"].nunique()}
        for m in SERIES_METRICS:
            mean, half = mean_ci(sub[m])
            row[m] = f"{mean:.3f} ± {half:.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def method_comparison(df, classes, rules):
    """Recalcula offline vários métodos de consenso sobre as mesmas probabilidades."""
    label_map = {c: i for i, c in enumerate(classes)}
    methods = {"local": None, "avg": (aggregate_avg, {})}
    for k in (1, 2, 3):
        r = copy.deepcopy(rules)
        r["min_nos_default"] = k
        for rule in r["regras"]:
            rule.pop("min_nos", None)
        methods[f"rules k={k}"] = (rules_aggregation, dict(rules_config=r, label_map=label_map, verbose=False))
    methods["rules (config)"] = (rules_aggregation, dict(rules_config=rules, label_map=label_map, verbose=False))

    rows = []
    for name, spec in methods.items():
        pred = df["local_pred"] if spec is None else recompute_consensus(df, spec[0], classes, **spec[1])
        for run_id, idx in df.groupby("run_id").groups.items():
            m = classification_metrics(df.loc[idx, "GLOBAL_STATUS"], pred.loc[idx], classes)
            rows.append({"method": name, "run_id": run_id, **m})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="+", help="pastas experiments/<data>_<nome>")
    parser.add_argument("--out", help="pasta de saída (default: <1ª experiência>/figures)")
    parser.add_argument("--run-filter", help="só runs cujo run_id contém este texto (ex: cenarioA)")
    args = parser.parse_args()

    df, rules, report = load_experiments(args.experiments)
    print("\n".join(report))
    if df.empty:
        sys.exit("❌ Sem dados com ground truth.")
    if args.run_filter:
        df = df[df["run_id"].str.contains(args.run_filter)]

    out = Path(args.out or Path(args.experiments[0]) / "figures")
    out.mkdir(parents=True, exist_ok=True)
    classes = class_list(rules, df)
    paper_style()

    print(f"\nRuns: {df.groupby(['mode', 'run_id']).size().to_dict()}")

    # --- Tabelas ---
    per_run = per_run_series(df, classes)
    per_run.to_csv(out / "metrics_per_run.csv", index=False)
    summary = summary_table(per_run)
    summary.to_csv(out / "metrics_summary.csv", index=False)
    print("\n" + summary.to_string(index=False))

    tests = []
    for mode, sub in df.groupby("mode"):
        t = mcnemar(sub["GLOBAL_STATUS"], sub["local_pred"], sub["global_pred"])
        tests.append({"mode": mode, "n": len(sub), **t})
    pd.DataFrame(tests).to_csv(out / "mcnemar_local_vs_consensus.csv", index=False)
    print("\nMcNemar local vs consenso:\n" + pd.DataFrame(tests).to_string(index=False))

    # --- Figuras ---
    made = [fig_local_vs_consensus(per_run, out), fig_corrections(df, classes, out), fig_latency(df, out)]
    for mode in df["mode"].unique():
        made.append(fig_confusion(df, classes, mode, out))
    if rules and "round_probs" in df.columns:
        methods = method_comparison(df, classes, rules)
        methods.to_csv(out / "consensus_methods.csv", index=False)
        made.append(fig_methods(methods, out))

    print("\n📊 Figuras:")
    for p in filter(None, made):
        print(f"   {p}")


if __name__ == "__main__":
    main()
