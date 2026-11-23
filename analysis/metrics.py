"""
Métricas de classificação (só numpy/scipy, sem sklearn).

Escolhidas para classes desequilibradas (quase tudo é SAFE):
  - macro_f1, balanced_accuracy, mcc: não são dominadas pela classe maioritária
  - danger_recall: dos instantes com perigo real (GT != SAFE), quantos foram detetados
    como perigo (qualquer classe de perigo). É a métrica de segurança.
"""
import numpy as np
import pandas as pd
from scipy import stats

SAFE = "SAFE"


def confusion(y_true, y_pred, classes):
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    return cm


def classification_metrics(y_true, y_pred, classes):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = confusion(y_true, y_pred, classes)
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)        # nº real de cada classe
    predicted = cm.sum(axis=0).astype(float)      # nº previsto de cada classe

    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, tp / support, np.nan)
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)

    present = support > 0  # só classes que existem no GT entram nas médias
    n = cm.sum()
    # MCC multiclasse (Gorodkin)
    c, s = tp.sum(), n
    cov_tp = c * s - (support * predicted).sum()
    denom = np.sqrt((s**2 - (predicted**2).sum()) * (s**2 - (support**2).sum()))

    danger_true = y_true != SAFE
    danger_pred = y_pred != SAFE
    out = {
        "n": int(n),
        "accuracy": c / n if n else np.nan,
        "balanced_accuracy": np.nanmean(recall[present]) if present.any() else np.nan,
        "macro_f1": np.mean(f1[present]) if present.any() else np.nan,
        "mcc": cov_tp / denom if denom > 0 else 0.0,
        "danger_recall": (danger_pred & danger_true).sum() / danger_true.sum() if danger_true.any() else np.nan,
        "false_alarm_rate": (danger_pred & ~danger_true).sum() / (~danger_true).sum() if (~danger_true).any() else np.nan,
    }
    for cls, r in zip(classes, recall):
        out[f"recall_{cls}"] = r
    return out


def mcnemar(y_true, pred_a, pred_b):
    """
    Teste de McNemar exato (binomial) para comparar 2 classificadores nas MESMAS amostras
    (ex: local vs consenso). n01 = só A acerta, n10 = só B acerta.
    """
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    n01, n10 = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    p = stats.binomtest(n01, n01 + n10, 0.5).pvalue if (n01 + n10) else 1.0
    return {"only_a_correct": n01, "only_b_correct": n10, "p_value": p}


def mean_ci(values, confidence=0.95):
    """Média e meia-largura do intervalo de confiança (t de Student) entre repetições."""
    values = np.asarray([v for v in values if not pd.isna(v)], dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return values[0], 0.0
    half = stats.t.ppf((1 + confidence) / 2, len(values) - 1) * stats.sem(values)
    return values.mean(), half
