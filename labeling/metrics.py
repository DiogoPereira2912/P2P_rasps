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

def load_inference_data(path, label_map=None):
    """Carrega os dados gravados pelo Agregador."""
    try:
        if path == "data_exports/local_inf_data_rasp1/":
            dt = DeltaTable(path)
            df = dt.to_pandas()
            df["ts"] = pd.to_datetime(df["ts"]).astype("datetime64[ns]")
            df = df.sort_values("ts")
            df["local_pred"] = df["local_pred"].map(label_map)
            return df
        else:
            dt = DeltaTable(path)
            df = dt.to_pandas()
            df["ts"] = pd.to_datetime(df["ts"]).astype("datetime64[ns]")
            df = df.sort_values("ts")
            return df

    except Exception as e:
        print(f"Erro ao carregar: {e}")
        return pd.DataFrame()


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
        self.df_local = load_inference_data(self.data_path_local, label_map=self.label_map)
        self.df_global = load_inference_data(self.data_path_global, label_map=self.label_map)

        merged_df = pd.merge_asof(
            self.df_local,
            self.df_global[["ts", self.df_global.columns[-1]]],
            on="ts",
            direction="nearest",
        )
        merged_df = merged_df.dropna()
        if merged_df.empty:
            print("⚠️ Nenhum dado para analisar.")
            return

        analyze_consensus_impact(merged_df, ground_truth_col="GLOBAL_STATUS")
        generate_classification_reports(
            merged_df,
            y_true_col="GLOBAL_STATUS",
            path=self.results_path,
            device_id=self.device_id,
        )
        plot_confusion_matrices(
            merged_df,
            y_true_col="GLOBAL_STATUS",
            label_map=self.label_map,
            path=self.results_path,
            device_id=self.device_id,
        )