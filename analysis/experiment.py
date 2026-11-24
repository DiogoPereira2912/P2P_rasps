"""
Carrega uma (ou várias) pastas experiments/<data>_<nome>/ trazidas com
tools/fetch_results.sh e junta cada predição ao seu ground truth.

Estrutura esperada:
    experiments/<exp>/<nó>/data_exports/local_inf_data_<nó>/   predições (aggregation)
    experiments/<exp>/<nó>/data_exports/inference_output_<nó>/ ground truth (labeler)
    experiments/<exp>/global_rules.yaml                        regras usadas no Pi
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from delta_io import read_delta

# Predição <-> ground truth: as duas linhas vêm do mesmo labeler com o mesmo ts,
# por isso devem coincidir; a tolerância só absorve arredondamentos.
GT_TOLERANCE = pd.Timedelta("100ms")

# Colunas acrescentadas ao local_inf_data nas versões novas do aggregation.
# Em dados antigos ficam com este valor, para não partir a análise.
CONTEXT_DEFAULTS = {"mode": "desconhecido", "agg_method": "desconhecido", "run_id": "sem_run_id"}


def _node_dirs(exp_dir):
    return [d for d in sorted(Path(exp_dir).iterdir()) if (d / "data_exports").is_dir()]


def load_predictions(exp_dir):
    frames = []
    for node_dir in _node_dirs(exp_dir):
        # Só a tabela do PRÓPRIO nó: os Pis podem ter cópias antigas de tabelas de outros
        # nós (ex: local_inf_data_rasp2 em todos), que não são deste run.
        # Em FL o central grava aqui as linhas de todos os nós (coluna "device").
        table = node_dir / "data_exports" / f"local_inf_data_{node_dir.name}"
        df = read_delta(table)
        if df.empty:
            continue
        df["device"] = df["device"].fillna(node_dir.name) if "device" in df.columns else node_dir.name
        df["experiment"] = Path(exp_dir).name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for col, default in CONTEXT_DEFAULTS.items():
        df[col] = df[col].fillna(default) if col in df.columns else default
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601").astype("datetime64[ns]")
    return df


def load_ground_truth(exp_dir):
    frames = []
    for node_dir in _node_dirs(exp_dir):
        table = node_dir / "data_exports" / f"inference_output_{node_dir.name}"
        df = read_delta(table)
        if df.empty or "GLOBAL_STATUS" not in df.columns:
            continue
        df = df[["ts", "GLOBAL_STATUS"]].copy()
        df["device"] = node_dir.name
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["ts", "GLOBAL_STATUS", "device"])
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601").astype("datetime64[ns]")
    return df.drop_duplicates(["device", "ts"])


def load_rules(exp_dir):
    path = Path(exp_dir) / "global_rules.yaml"
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text())


def join_ground_truth(pred, gt, tolerance=GT_TOLERANCE):
    """
    Cada predição do nó X recebe o GLOBAL_STATUS do ground truth do nó X com o mesmo ts.
    Devolve (dataframe com GT, nº de predições descartadas por não terem GT).
    """
    pred = pred.sort_values("ts")
    gt = gt.sort_values("ts")
    merged = pd.merge_asof(pred, gt, on="ts", by="device", direction="nearest", tolerance=tolerance)
    matched = merged.dropna(subset=["GLOBAL_STATUS"]).copy()
    return matched, len(merged) - len(matched)


def load_experiments(exp_dirs):
    """Junta várias pastas (ex: repetições). Devolve (df com GT, regras, resumo de carregamento)."""
    all_df, rules, report = [], None, []
    for exp_dir in exp_dirs:
        pred, gt = load_predictions(exp_dir), load_ground_truth(exp_dir)
        rules = rules or load_rules(exp_dir)
        if pred.empty:
            report.append(f"{Path(exp_dir).name}: sem predições")
            continue
        matched, dropped = join_ground_truth(pred, gt)
        report.append(
            f"{Path(exp_dir).name}: {len(matched)}/{len(pred)} predições com ground truth "
            f"(descartadas: {dropped})"
        )
        all_df.append(matched)
    df = pd.concat(all_df, ignore_index=True) if all_df else pd.DataFrame()
    return df, rules, report


def recompute_consensus(df, aggregate_fn, label_names, **kwargs):
    """
    Recalcula offline a decisão global com outro método/parâmetros, sobre EXATAMENTE as
    mesmas probabilidades dos nós (coluna round_probs). Permite comparar métodos
    (ex: avg vs rules, min_nos 1 vs 2) sem repetir a experiência nos Pis.
    """
    if "round_probs" not in df.columns:
        raise ValueError("Dados sem round_probs (versão antiga do aggregation).")
    cache = {}
    decisions = []
    for raw in df["round_probs"]:
        if not isinstance(raw, str):  # linha sem probabilidades dos nós
            decisions.append(None)
            continue
        if raw not in cache:
            probs = {k: [v] for k, v in json.loads(raw).items()}
            out = np.asarray(aggregate_fn(probs, **kwargs))
            cache[raw] = label_names[int(np.argmax(out[0]))]
        decisions.append(cache[raw])
    return pd.Series(decisions, index=df.index)


# ---------------------------------------------------------------------------
# Ground truth humano (tools/event_logger.py)
# ---------------------------------------------------------------------------
def load_event_logs(paths):
    """Junta os registos de eventos (CSV do event_logger); ignora as notas."""
    frames = [pd.read_csv(p) for p in paths]
    if not frames:
        return pd.DataFrame()
    ev = pd.concat(frames, ignore_index=True)
    ev = ev[ev["acao"] != "NOTA"].copy()
    # Os ts das predições vêm de pd.to_datetime(..., unit="s") nos Pis (UTC, sem fuso):
    # converter da mesma forma para alinharem.
    ev["inicio"] = pd.to_datetime(ev["inicio_ts"], unit="s")
    ev["fim"] = pd.to_datetime(ev["fim_ts"], unit="s")
    return ev.sort_values("inicio")


def attach_human_gt(df, events):
    """
    Acrescenta HUMAN_GT (estado esperado do shop floor segundo o guião) e ACAO a cada
    predição cujo instante cai dentro de um segmento registado do run com a mesma etiqueta.
    O run_id é <data>_<GL|FL>_<etiqueta>; a etiqueta do registo tem de ser a mesma.
    """
    df = df.copy()
    df["HUMAN_GT"], df["ACAO"] = None, None
    if events.empty:
        return df
    for tag, ev in events.groupby("tag"):
        mask = df["run_id"].astype(str).str.endswith(f"_{tag}")
        if not mask.any():
            continue
        sub = df.loc[mask].sort_values("ts")
        m = pd.merge_asof(sub[["ts"]].reset_index(), ev[["inicio", "fim", "esperado", "acao"]],
                          left_on="ts", right_on="inicio", direction="backward")
        inside = m["ts"] < m["fim"]
        idx = m.loc[inside, "index"]
        df.loc[idx, "HUMAN_GT"] = m.loc[inside, "esperado"].values
        df.loc[idx, "ACAO"] = m.loc[inside, "acao"].values
    return df
