import os, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from labeling_utils import load_class_mappings
from deltalake import DeltaTable

GLOBAL_RULES_PATH = "labeling/global_rules.yaml"

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 12})

# Janela máxima para associar uma predição ao seu ground truth. As linhas de
# local_inf_data_<nó> e inference_output_<nó> vêm do mesmo labeler, com o mesmo ts,
# por isso o normal é coincidirem exatamente; a tolerância só absorve arredondamentos.
GT_MATCH_TOLERANCE = pd.Timedelta("100ms")


def load_inference_data(path):
    """Carrega uma tabela Delta (predições do Agregador ou ground truth do Labeler)."""
    try:
        df = DeltaTable(path).to_pandas()
        df["ts"] = pd.to_datetime(df["ts"], format="ISO8601").astype("datetime64[ns]")
        return df.sort_values("ts")
    except Exception as e:
        print(f"Erro ao carregar {path}: {e}")
        return pd.DataFrame()


def match_predictions_to_ground_truth(df_pred, df_gt, tolerance=GT_MATCH_TOLERANCE):
    """
    Associa cada predição ao GLOBAL_STATUS (ground truth) com o mesmo timestamp.
    Predições sem ground truth dentro da tolerância são descartadas e contadas,
    para o número poder ser reportado (ex: no artigo).
    """
    merged = pd.merge_asof(
        df_pred,
        df_gt[["ts", "GLOBAL_STATUS"]],
        on="ts",
        direction="nearest",
        tolerance=tolerance,
    )
    matched = merged.dropna(subset=["GLOBAL_STATUS"])
    n_dropped = len(merged) - len(matched)
    print(
        f"🔗 Predições com ground truth: {len(matched)}/{len(merged)} "
        f"(descartadas: {n_dropped}, tolerância {tolerance})"
    )
    return matched


def analyze_consensus_impact(df, ground_truth_col=None):
    """
    Analisa quantas vezes a decisão global foi diferente da local.
    """
    total = len(df)

    divergences = df[df["local_pred"] != df["global_pred"]]
    n_divergences = len(divergences)

    print(f"--- Impacto do Consenso ---")
    print(f"Total de Inferências: {total}")
    print(f"Intervenções da Rede: {n_divergences} ({n_divergences/total:.1%})")

    if ground_truth_col and ground_truth_col in df.columns:
        saved_by_network = divergences[
            divergences["global_pred"] == divergences[ground_truth_col]
        ]
        ruined_by_network = divergences[
            divergences["local_pred"] == divergences[ground_truth_col]
        ]

        print(f"✅ Salvo pela Rede (Correções): {len(saved_by_network)}")
        print(f"❌ Estragado pela Rede (Regressões): {len(ruined_by_network)}")

    return divergences


def generate_classification_reports(
    df, y_true_col="GLOBAL_STATUS", path="results/", device_id="unknown"
):
    """Gera os relatórios de classificação (F1, Precision, Recall) e guarda em ficheiro de texto."""

    if y_true_col not in df.columns:
        return

    y_true = df[y_true_col].astype(str)
    y_local = df["local_pred"].astype(str)
    y_global = df["global_pred"].astype(str)

    report_local = classification_report(y_true, y_local, zero_division=0)
    report_global = classification_report(y_true, y_global, zero_division=0)

    if not os.path.exists(path):
        os.makedirs(path)

    filename = f"{path}classification_report_{device_id}_{time.strftime('%d_%H_%M_%S')}.txt"

    with open(filename, "w") as f:
        f.write(f"--- RELATÓRIO DO DISPOSITIVO: {device_id} ---\n\n")
        f.write("=== PERFORMANCE DO MODELO LOCAL ===\n")
        f.write(report_local)
        f.write("\n\n=== PERFORMANCE APÓS CONSENSO GLOBAL ===\n")
        f.write(report_global)

    print(f"📄 Relatório de classificação exportado: {filename}")


def plot_confusion_matrices(
    df, y_true_col="GLOBAL_STATUS", label_map=None, path=None, device_id=None
):
    """
    Plota duas matrizes lado a lado: Performance Local vs Performance Global.
    """
    if y_true_col not in df.columns:
        print("⚠️ Sem Ground Truth, não é possível gerar Matriz de Confusão.")
        return

    classes = sorted(list(label_map.values()))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    cm_local = confusion_matrix(df[y_true_col], df["local_pred"], labels=classes)
    sns.heatmap(
        cm_local,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0],
        xticklabels=classes,
        yticklabels=classes,
    )
    axes[0].set_title("Matriz de Confusão: Modelo Local")

    cm_global = confusion_matrix(df[y_true_col], df["global_pred"], labels=classes)
    sns.heatmap(
        cm_global,
        annot=True,
        fmt="d",
        cmap="Greens",
        ax=axes[1],
        xticklabels=classes,
        yticklabels=classes,
    )
    axes[1].set_title("Matriz de Confusão: Consenso Global")

    if not os.path.exists(path):
        os.makedirs(path)

    filename = f"{path}inference_cm_batch_{device_id}_{time.strftime('%d_%H_%M_%S')}.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Gráfico exportado com sucesso: {filename}")


class MetricsAnalyzer:
    def __init__(
        self, data_path_local, data_path_global, results_path="results/", device_id=None
    ):
        self.data_path_local = data_path_local
        self.data_path_global = data_path_global
        self.device_id = device_id
        _, self.label_map, _ = load_class_mappings(GLOBAL_RULES_PATH)
        self.results_path = results_path
        self.df = None

    def run_analysis(self):
        # data_path_local  -> predições (local_inf_data_<nó>, gravado pelo Agregador)
        # data_path_global -> ground truth (inference_output_<nó>, gravado pelo Labeler)
        self.df_local = load_inference_data(self.data_path_local)
        self.df_global = load_inference_data(self.data_path_global)

        if self.df_local.empty or "GLOBAL_STATUS" not in self.df_global.columns:
            print("⚠️ Faltam predições ou ground truth para analisar.")
            return

        merged_df = match_predictions_to_ground_truth(self.df_local, self.df_global)
        if merged_df.empty:
            print("⚠️ Nenhuma predição com ground truth correspondente.")
            return

        # Um relatório por modo (GL / FL): os dois ficam na mesma tabela Delta.
        # As figuras do artigo são feitas no Mac (analysis/make_figures.py).
        groups = merged_df.groupby("mode") if "mode" in merged_df.columns else [("todos", merged_df)]
        for mode, sub in groups:
            print(f"\n===== Modo: {mode} ({len(sub)} predições) =====")
            tag = f"{self.device_id}_{mode}"
            analyze_consensus_impact(sub, ground_truth_col="GLOBAL_STATUS")
            generate_classification_reports(
                sub,
                y_true_col="GLOBAL_STATUS",
                path=self.results_path,
                device_id=tag,
            )
            plot_confusion_matrices(
                sub,
                y_true_col="GLOBAL_STATUS",
                label_map=self.label_map,
                path=self.results_path,
                device_id=tag,
            )
