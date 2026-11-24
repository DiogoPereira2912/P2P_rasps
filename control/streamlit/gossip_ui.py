import streamlit as st
import os, sys, time, glob, subprocess
import pandas as pd
from yaml import Loader, load
from mqtt_com import Processor

# --- CONFIGURAÇÃO DEFAULT ---
# Raiz do repo: no contentor está montada em /repo (ver control/compose.yaml).
# Fora do Docker (streamlit run a partir de control/streamlit) fica dois níveis acima.
REPO_PATH = os.getenv("REPO_PATH", "../..")
INVENTORY_PATH = os.path.join(REPO_PATH, "inventory.yaml")
DEPLOY_SCRIPT = os.path.join(REPO_PATH, "tools", "deploy.sh")  # código + config -> Pis

# Métricas de sistema por run (lidas do Prometheus): mesmo código que tools/export_metrics.py
sys.path.insert(0, os.path.join(REPO_PATH, "tools"))
import importlib
import export_metrics as em

# O Streamlit volta a correr este ficheiro a cada interação mas mantém os módulos importados
# em memória: recarregar para apanhar alterações em tools/export_metrics.py sem reiniciar a UI.
em = importlib.reload(em)
TAR_DIRECTORY = "/app/imgs"  # Diretoria onde estão os ficheiros .tar para deploy
NODE_SCRIPT = "../node.sh"  # Ações SSH nos Pis (start / load)


def load_inventory(path):
    """
    Lê o inventory.yaml e devolve:
        - node_names: {ip: nome} para mostrar "rasp1 (192.168.0.110)" na UI
        - mqtt_port: porta MQTT comum a todos os nós
    """
    if not os.path.exists(path):
        return {}, 1884
    with open(path, "r") as file:
        inventory = load(file, Loader=Loader)
    node_names = {node["ip"]: name for name, node in inventory["nodes"].items()}
    mqtt_port = inventory["defaults"].get("mosquitto_port", 1884)
    return node_names, mqtt_port


NODE_NAMES, DEFAULT_PORT = load_inventory(INVENTORY_PATH)
DEFAULT_IDS = ", ".join(NODE_NAMES.keys())
# DEFAULT_IDS = "100.117.56.27, 100.87.185.23, 100.118.159.26" # used with tailscale


def node_label(ip):
    """Texto mostrado nas listas de seleção: 'rasp1 (192.168.0.110)' ou só o IP."""
    return f"{NODE_NAMES[ip]} ({ip})" if ip in NODE_NAMES else ip

st.set_page_config(page_title="AIDI Control Tower", layout="wide", page_icon="🗼")

# --- HEADER ---
st.title("🗼 AIDI Gossip Control Tower")
st.markdown("---")

# --- CONFIGURAÇÃO DE REDE ---
st.sidebar.header("📡 Configuração de Rede")
if not NODE_NAMES:
    st.sidebar.warning(f"⚠️ inventory.yaml não encontrado em '{INVENTORY_PATH}'.")
rasp_ids_input = st.sidebar.text_area("Raspberry IPs", value=DEFAULT_IDS, height=100)
port = st.sidebar.number_input(
    "Porta MQTT", min_value=1, max_value=65535, value=DEFAULT_PORT
)

raspberry_ids = [r.strip() for r in rasp_ids_input.split(",") if r.strip()]

# --- CONFIGURAÇÃO GLOBAL DE PARÂMETROS ---
st.sidebar.header("⚙️ Parâmetros de Recolha")
save_interval = st.sidebar.number_input("Intervalo (s)", min_value=1, value=5)
max_saves = st.sidebar.number_input("Batches Máximos", min_value=1, value=10)

# --- CONFIGURAÇÃO PARÂMETROS INFERENCE MODE ---
st.sidebar.header("⚙️ Parâmetros de Inferência")
max_inferences = st.sidebar.number_input("Inferências por batch", min_value=1, value=10)


# --- FUNÇÃO DE BROADCAST ---
def send_command_to_all(command, payload, raspberry_ids, port):
    results = {}
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, rasp_ip in enumerate(raspberry_ids):
        status_text.text(f"A conectar a {rasp_ip}...")
        producer = None
        try:
            producer = Processor(
                raspberry_id=f"cmd_{int(time.time())}",
                broker=rasp_ip,
                port=port,
                base_topic="",
                qos=1,
            )
            msg = {"command": command, **payload}
            producer.publish(msg, topic="system/control/")

            results[rasp_ip] = "✅ OK"
        except Exception as e:
            results[rasp_ip] = f"❌ Erro: {e}"
        finally:
            if producer:
                try:
                    if hasattr(producer, "disconnect"):
                        producer.disconnect()
                    elif hasattr(producer, "client"):
                        producer.client.disconnect()
                except:
                    pass

        progress_bar.progress((i + 1) / len(raspberry_ids))
        time.sleep(0.1)  # Pequeno delay para não congestionar a rede local

    status_text.empty()
    progress_bar.empty()
    return results


# --- RESULTADOS (tab 5) ---
@st.cache_data(ttl=30, show_spinner=False)
def cached_runs(since):
    return em.find_runs(em.parse_duration(since))


@st.cache_data(show_spinner="A calcular métricas do run...")
def cached_kpis(run_id, mode, start, end):
    """Um run já terminado não muda: fica em cache (chave = run + intervalo)."""
    run = {"run_id": run_id, "mode": mode, "start": start, "end": end}
    return em.kpis(em.run_metrics(run), em.run_timeseries(run))


def run_repo_command(cmd, label):
    """Corre um script do repo (tools/, analysis/) e mostra o output. Devolve True se correu bem."""
    result = subprocess.run(cmd, cwd=REPO_PATH, capture_output=True, text=True)
    with st.expander(f"Logs: {label}", expanded=result.returncode != 0):
        st.code((result.stdout + result.stderr)[-4000:] or "(sem output)")
    if result.returncode != 0:
        if "matplotlib" in result.stderr or "scipy" in result.stderr:
            st.error("Faltam bibliotecas na imagem da UI: `cd control && docker compose up -d --build`.")
        return False
    return True


def show_classification_gains(summary):
    """Cartões local -> consenso (GL e FL) a partir do metrics_summary.csv ("média ± ic")."""
    value = lambda cell: float(str(cell).split("±")[0])
    rows = summary.set_index("series")
    if "local" not in rows.index:
        return
    st.markdown("**🎯 Ganho do consenso face ao nó isolado** (contra o ground truth do shop floor)")
    labels = {"macro_f1": "Macro-F1", "danger_recall": "Deteção de perigos (recall)",
              "false_alarm_rate": "Falsos alarmes"}
    cols = st.columns(len(labels))
    for col, (metric, label) in zip(cols, labels.items()):
        local = value(rows.loc["local", metric])
        parts = [f"{name} {value(rows.loc[mode, metric]):.2f}" for mode, name in
                 (("gossip", "GL"), ("federated", "FL")) if mode in rows.index]
        best = max((value(rows.loc[m, metric]) for m in ("gossip", "federated") if m in rows.index), default=local)
        delta = best - local
        col.metric(label, f"local {local:.2f} → " + " · ".join(parts), f"{delta:+.2f}",
                   delta_color="inverse" if metric == "false_alarm_rate" else "normal")


def show_figures(rel_dir, where_text):
    """Mostra as figuras PNG geradas (os PDF para o artigo ficam na mesma pasta)."""
    pngs = sorted(glob.glob(os.path.join(REPO_PATH, rel_dir, "*.png")))
    st.success(f"✅ {len(pngs)} figuras (PDF + PNG). {where_text}")
    cols = st.columns(2)
    for i, png in enumerate(pngs):
        cols[i % 2].image(png, caption=os.path.basename(png).replace(".png", ""), width="stretch")


# --- INTERFACE PRINCIPAL (TABS) ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🛠️ 1. Setup & Deploy",
        "📂 2. Data Collection",
        "🧠 3. Treino (Offline)",
        "🎯 4. Inferência (Live)",
        "📊 5. Test and Metrics Generation",
    ]
)

# --- TAB 1: SETUP & DEPLOY ---
with tab1:
    st.header("🟢 Inicialização da Frota")
    st.info("Levanta os serviços (ex: docker-compose up -d) nos nós configurados.")

    startup_ips = st.multiselect(
        "Seleciona os nós para levantar o serviço:",
        options=raspberry_ids,
        default=raspberry_ids,
        format_func=node_label,
        key="startup_select",  # Key única para não dar conflito com o multiselect do deploy
    )

    if st.button("🔌 Levantar Serviço", type="primary", width="stretch"):
        if not startup_ips:
            st.warning("⚠️ Seleciona pelo menos um Raspberry Pi para iniciar.")
        else:
            with st.status(
                f"A iniciar serviços em {len(startup_ips)} nós...", expanded=True
            ) as status:
                for ip in startup_ips:
                    st.write(f"⏳ A levantar serviço em **{ip}**...")
                    try:
                        result = subprocess.run(
                            ["bash", NODE_SCRIPT, "start", ip],
                            capture_output=True,
                            text=True,
                            check=True,
                        )
                        st.success(f"✅ {ip}: Serviço online!")
                        with st.expander(f"Ver logs ({ip})"):
                            st.code(result.stdout)
                    except subprocess.CalledProcessError as e:
                        st.error(f"❌ {ip}: Falha ao iniciar.")
                        with st.expander(f"Ver erros ({ip})"):
                            st.code(e.stderr)

                status.update(
                    label="Processo de inicialização concluído!",
                    state="complete",
                )

    st.markdown("---")
    st.header("📤 Atualizar Código")
    st.info(
        "Envia o código e a config de cada nó (gerada a partir do inventory.yaml). "
        "Não toca em data_exports/ nem models/ nos Pis."
    )

    update_ips = st.multiselect(
        "Seleciona os nós a atualizar:",
        options=raspberry_ids,
        default=raspberry_ids,
        format_func=node_label,
        key="update_select",
    )
    restart_after = st.checkbox(
        "Reiniciar os serviços Python no fim (peer, pipeline, aggregation, labeler, detector)",
        value=True,
    )

    if st.button("📤 Atualizar Código", type="primary", width="stretch"):
        # O deploy.sh trabalha com os nomes do inventário (config gerada por nó),
        # por isso IPs que não estão no inventory.yaml não podem ser atualizados.
        sem_inventario = [ip for ip in update_ips if ip not in NODE_NAMES]
        if not update_ips:
            st.warning("⚠️ Seleciona pelo menos um Raspberry Pi.")
        elif sem_inventario:
            st.error(
                f"❌ Estes IPs não estão no inventory.yaml: {', '.join(sem_inventario)}"
            )
        else:
            with st.status(
                f"A atualizar código em {len(update_ips)} nós...", expanded=True
            ) as status:
                for ip in update_ips:
                    st.write(f"⏳ A atualizar **{node_label(ip)}**...")
                    cmd = ["bash", DEPLOY_SCRIPT, NODE_NAMES[ip]]
                    if restart_after:
                        cmd.append("--restart")
                    try:
                        result = subprocess.run(
                            cmd, capture_output=True, text=True, check=True
                        )
                        st.success(f"✅ {node_label(ip)}: Código atualizado!")
                        with st.expander(f"Ver logs ({ip})"):
                            st.code(result.stdout)
                    except subprocess.CalledProcessError as e:
                        st.error(f"❌ {node_label(ip)}: Falha na atualização.")
                        with st.expander(f"Ver erros ({ip})"):
                            st.code(e.stdout + e.stderr)

                status.update(label="Atualização concluída!", state="complete")

    st.markdown("---")
    st.header("🛠️ Deploy de Imagens Docker")
    st.info("Transfere e carrega (docker load) as imagens nos nós selecionados.")

    col1, col2 = st.columns(2)

    with col1:
        target_ips = st.multiselect(
            "Seleciona os Raspberries alvo para o deploy:",
            options=raspberry_ids,
            default=raspberry_ids,
            format_func=node_label,
        )

    with col2:
        search_pattern = os.path.join(TAR_DIRECTORY, "*.tar")
        ficheiros_tar_paths = glob.glob(search_pattern)

        nomes_tar = [os.path.basename(p) for p in ficheiros_tar_paths]

        if not nomes_tar:
            nomes_tar = ["Nenhum ficheiro .tar encontrado"]

        tar_selecionado_nome = st.selectbox(
            "Ficheiro Docker (.tar) a enviar:",
            nomes_tar,
        )

    if st.button("🚀 Deploy Images", type="primary", width="stretch"):

        if not target_ips:
            st.warning("⚠️ Seleciona pelo menos um Raspberry Pi para continuar.")
        elif tar_selecionado_nome == "Nenhum ficheiro .tar encontrado":
            st.error(
                f"❌ Não há ficheiros .tar na pasta configurada ('{TAR_DIRECTORY}')."
            )
        else:
            # Junta o caminho da pasta com o nome do ficheiro para passar ao bash
            tar_completo = os.path.join(TAR_DIRECTORY, tar_selecionado_nome)
            with st.status(
                f"A iniciar deploy para {len(target_ips)} nós...", expanded=True
            ) as status:
                for ip in target_ips:
                    st.write(f"⏳ A enviar '{tar_selecionado_nome}' para **{ip}**...")

                    try:
                        result = subprocess.run(
                            ["bash", NODE_SCRIPT, "load", ip, tar_completo],
                            capture_output=True,
                            text=True,
                            check=True,
                        )
                        st.success(f"✅ {ip}: Deploy concluído com sucesso!")
                        with st.expander(f"Ver logs do terminal ({ip})"):
                            st.code(result.stdout)
                    except subprocess.CalledProcessError as e:
                        st.error(f"❌ {ip}: Falha no deploy.")
                        with st.expander(f"Ver logs de erro ({ip})"):
                            st.code(e.stderr)

                status.update(
                    label="Deploy finalizado em todos os nós selecionados!",
                    state="complete",
                )

# --- TAB 2: DATA COLLECTION ---

with tab2:
    st.header("Recolha de Dados")
    st.info("Neste modo, os Raspberries guardam os dados brutos (Raw) no disco local.")

    col_start, col_stop = st.columns(2)

    with col_start:
        if st.button(
            "🔴 INICIAR GRAVAÇÃO",
            type="primary",
            width="stretch",
            key="btn_col_start",
        ):
            # CORREÇÃO: Passar a config aqui!
            config_payload = {
                "config": {"save_interval": save_interval, "max_saves": max_saves}
            }
            with st.status("A enviar ordem de GRAVAÇÃO...", expanded=True):
                res = send_command_to_all(
                    "COLLECTION", config_payload, raspberry_ids, port
                )
                for ip, status in res.items():
                    st.write(f"**{ip}**: {status}")
            st.toast("Modo DATA COLLECTION Ativado!")

    with col_stop:
        if st.button(
            "⏹ PARAR GRAVAÇÃO",
            type="secondary",
            width="stretch",
            key="btn_col_stop",
        ):
            send_command_to_all("STOP", {}, raspberry_ids, port)
            st.toast("Paragem enviada!")

# --- TAB 3: TREINO ---
with tab3:
    st.header("Treino do Modelo")
    st.warning("⚠️ O Treino consome muito CPU. A deteção de vídeo pode ficar lenta.")

    col_start, col_stop = st.columns(2)

    with col_start:
        if st.button(
            "🏋️ INICIAR TREINO",
            type="primary",
            width="stretch",
            key="btn_train_start",
        ):
            with st.status("A enviar ordem de TREINO...", expanded=True):
                res = send_command_to_all("TRAIN", {}, raspberry_ids, port)
                for ip, status in res.items():
                    st.write(f"**{ip}**: {status}")
            st.toast("Modo TRAINING Ativado!")

    with col_stop:
        if st.button(
            "⏹ PARAR TREINO",
            type="secondary",
            width="stretch",
            key="btn_train_stop",
        ):
            send_command_to_all("STOP", {}, raspberry_ids, port)
            st.toast("Treino Cancelado!")

# --- TAB 4: INFERÊNCIA ---
with tab4:
    st.header("Inferência em Tempo Real")
    st.success("O sistema classifica o perigo localmente e roteia os dados pela rede.")

    test_mode = st.radio(
        "Arquitetura de Roteamento:",
        ["Gossip Learning (GL)", "Federated Learning (FL)"],
    )

    central_ip_selecionado = None
    if "Federated" in test_mode:
        central_ip_selecionado = st.selectbox(
            "👑 Servidor Central (Debug/Orquestrador):",
            options=raspberry_ids,
            format_func=node_label,
            help="Este é o IP que vai receber os modelos e fazer a agregação no FL.",
        )
        st.info(
            f"Todo o tráfego da rede será roteado para **{central_ip_selecionado}**."
        )
    else:
        st.info(
            "🔄 Em Gossip, todos os nós comunicam entre si P2P. Não há servidor central."
        )

    # Etiqueta da experiência: vai em todas as linhas gravadas (run_id) para separar
    # GL vs FL, repetições, cenários... nos gráficos (analysis/).
    experiment_tag = st.text_input(
        "🏷️ Etiqueta da experiência (opcional)",
        placeholder="ex: cenario_A_rep1",
        help="O run_id final fica <data_hora>_<GL|FL>_<etiqueta>.",
    )

    col_start, col_stop = st.columns(2)

    with col_start:
        if st.button(
            "🚀 INICIAR INFERÊNCIA",
            type="primary",
            width="stretch",
            key="btn_inf_start",
        ):

            modo_selecionado = "FL" if "Federated" in test_mode else "GL"

            run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{modo_selecionado}"
            if experiment_tag.strip():
                run_id += f"_{experiment_tag.strip().replace(' ', '_')}"

            config_payload = {
                "config": {
                    "mode": modo_selecionado,
                    "max_saves": max_inferences,
                    "run_id": run_id,
                }
            }

            if central_ip_selecionado:
                config_payload["config"]["central_ip"] = central_ip_selecionado

            with st.status(f"A ativar frota em modo {test_mode}...", expanded=True):
                res = send_command_to_all(
                    "INFERENCE", config_payload, raspberry_ids, port
                )
                for ip, status in res.items():
                    if ip == central_ip_selecionado:
                        st.write(f"👑 **{ip}** (Servidor): {status}")
                    else:
                        st.write(f"**{ip}**: {status}")

            st.success(f"🏷️ run_id: `{run_id}`")
            st.toast("Modo INFERENCE Ativado!")

    with col_stop:
        if st.button(
            "⏹ PARAR SISTEMA",
            type="secondary",
            width="stretch",
            key="btn_inf_stop",
        ):
            send_command_to_all("STOP", {}, raspberry_ids, port)
            st.error("Kill Switch Enviado!")

# --- TAB 5: TEST AND METRICS GENERATION ---
with tab5:
    st.header("Test and Metrics Generation")
    st.success("O sistema classifica em tempo real e gera métricas de desempenho.")

    col_start, col_stop = st.columns(2)

    with col_start:
        if st.button(
            "🚀 METRICS GENERATION",
            type="primary",
            width="stretch",
            key="btn_test_start",
        ):
            with st.status("A lançar fase de teste...", expanded=True):
                res = send_command_to_all("METRICS", {}, raspberry_ids, port)
                for ip, status in res.items():
                    st.write(f"**{ip}**: {status}")
            st.toast("Modo METRICS Ativado!")

    with col_stop:
        if st.button(
            "⏹ PARAR SISTEMA",
            type="secondary",
            width="stretch",
            key="btn_test_stop",
        ):
            send_command_to_all("STOP", {}, raspberry_ids, port)
            st.error("Kill Switch Enviado!")

    # ------------------------------------------------------------------
    # Resultados de sistema (Prometheus) — artigo de desempenho GL vs FL
    # ------------------------------------------------------------------
    st.markdown("---")
    st.header("📊 Resultados de sistema (GL vs FL)")
    st.caption(
        "Lidos do Prometheus, dentro da janela de cada run (do INFERENCE ao STOP). "
        "Os Pis não precisam de estar ligados."
    )

    since = st.selectbox("Procurar runs nas últimas", ["6h", "24h", "3d", "7d"], index=1)
    try:
        runs = cached_runs(since)
    except Exception as e:
        runs = []
        st.error(f"❌ Sem ligação ao Prometheus: {e}")

    if not runs:
        st.info("Nenhum run encontrado neste intervalo.")
    else:
        run_labels = {
            r["run_id"]: f"{r['run_id']}  ({r['mode']}, {r['end'] - r['start']:.0f} s)" for r in runs
        }
        selected = st.multiselect(
            "Runs", options=list(run_labels), default=list(run_labels), format_func=run_labels.get
        )
        chosen = [r for r in runs if r["run_id"] in selected]

        if chosen:
            table = pd.DataFrame([cached_kpis(r["run_id"], r["mode"], r["start"], r["end"]) for r in chosen])
            st.dataframe(table, hide_index=True, width="stretch")

            # Comparação rápida GL vs FL (média dos runs selecionados)
            by_mode = table.groupby("modo").mean(numeric_only=True)
            first = table.groupby("modo").first()
            if {"gossip", "federated"}.issubset(by_mode.index):
                gl, fl = by_mode.loc["gossip"], by_mode.loc["federated"]
                c1, c2, c3 = st.columns(3)
                c1.metric("Mensagens de consenso/min (GL vs FL)",
                          f"{gl['msgs consenso/min']:.0f} vs {fl['msgs consenso/min']:.0f}",
                          f"GL = {gl['msgs consenso/min'] / max(fl['msgs consenso/min'], 1):.1f}× FL",
                          delta_color="off")
                c2.metric("Latência decisão p50 (GL vs FL)",
                          f"{gl['latência decisão p50 (s)']:.2f} vs {fl['latência decisão p50 (s)']:.2f} s")
                c3.metric("CPU consenso máx/mín nó (GL | FL)",
                          f"{gl['CPU consenso máx nó (%)']:.1f}/{gl['CPU consenso mín nó (%)']:.1f} | "
                          f"{fl['CPU consenso máx nó (%)']:.1f}/{fl['CPU consenso mín nó (%)']:.1f} %",
                          "FL concentra no central", delta_color="off")

                # Complemento: o que o GL ganha em troca das mensagens a mais
                st.markdown("**🛡️ Robustez e distribuição** (o que o GL ganha em troca das mensagens a mais)")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Nós que decidem", f"GL {first.loc['gossip', 'nós que decidem']}",
                          f"FL {first.loc['federated', 'nós que decidem']}", delta_color="off")
                c2.metric("Ponto único de falha", f"GL: {first.loc['gossip', 'ponto único de falha']}",
                          f"FL: {first.loc['federated', 'ponto único de falha']}", delta_color="off")
                c3.metric("Desequilíbrio de carga (máx/mín)",
                          f"GL {gl['desequilíbrio de carga (máx/mín)']:.1f}×",
                          f"FL {fl['desequilíbrio de carga (máx/mín)']:.1f}× (central)", delta_color="off")
                c4.metric("Consenso alterou a decisão local",
                          f"GL {gl['consenso alterou local (%)']:.0f}% · FL {fl['consenso alterou local (%)']:.0f}%",
                          help="Rondas em que a decisão global foi DIFERENTE da predição local do nó. "
                               "Não diz se a mudança acertou: isso vê-se contra o ground truth, na secção "
                               "de classificação abaixo. Com todos os nós ativos, GL e FL decidem o mesmo, "
                               "por isso este valor deve ser parecido nos dois modos.")

            if st.button("📥 Exportar métricas + figuras de sistema", type="primary",
                         width="stretch", key="btn_export_system"):
                out_rel = f"experiments/system_metrics/{time.strftime('%Y-%m-%d_%H%M%S')}"
                with st.status("A exportar do Prometheus e a gerar figuras...", expanded=True) as status:
                    ok = run_repo_command(
                        [sys.executable, "tools/export_metrics.py", "--since", since,
                         "--run-ids", ",".join(selected), "--out", out_rel], "Export")
                    if ok:
                        ok = run_repo_command(
                            [sys.executable, "analysis/system_figures.py", out_rel], "Figuras")
                    status.update(label="Concluído" if ok else "Falhou", state="complete" if ok else "error")
                if ok:
                    show_figures(out_rel + "/figures", f"No Mac: P2P_rasps/{out_rel}/")

    # ------------------------------------------------------------------
    # Resultados de classificação (dados dos Pis) — artigo do caso de uso
    # ------------------------------------------------------------------
    st.markdown("---")
    st.header("🎯 Resultados de classificação (local vs consenso)")
    st.caption(
        "Traz data_exports/ de todos os Pis (tools/fetch_results.sh) e gera as figuras "
        "(analysis/make_figures.py). Os Pis têm de estar ligados."
    )
    col_a, col_b = st.columns(2)
    fetch_name = col_a.text_input("Nome da experiência", value="sessao", key="fetch_name")
    run_filter = col_b.text_input("Filtrar runs (texto do run_id, opcional)", key="fetch_filter",
                                  placeholder="ex: cenarioA")

    if st.button("📥 Trazer dados dos Pis + figuras de classificação", type="primary",
                 width="stretch", key="btn_export_clf"):
        with st.status("A trazer dados dos Pis...", expanded=True) as status:
            result = subprocess.run(["bash", "tools/fetch_results.sh", fetch_name.strip() or "sessao"],
                                    cwd=REPO_PATH, capture_output=True, text=True)
            with st.expander("Logs do fetch"):
                st.code(result.stdout + result.stderr)
            exp_dirs = [line.split("Resultados em:")[-1].strip()
                        for line in result.stdout.splitlines() if "Resultados em:" in line]
            ok = bool(exp_dirs)
            if ok:
                cmd = [sys.executable, "analysis/make_figures.py", exp_dirs[0]]
                if run_filter.strip():
                    cmd += ["--run-filter", run_filter.strip()]
                ok = run_repo_command(cmd, "Figuras")
            status.update(label="Concluído" if ok else "Falhou", state="complete" if ok else "error")
        if ok:
            summary = os.path.join(REPO_PATH, exp_dirs[0], "figures", "metrics_summary.csv")
            if os.path.exists(summary):
                show_classification_gains(pd.read_csv(summary))
                st.subheader("Resumo (média ± IC 95% entre runs)")
                st.dataframe(pd.read_csv(summary), hide_index=True, width="stretch")
            show_figures(exp_dirs[0] + "/figures", f"No Mac: P2P_rasps/{exp_dirs[0]}/")

st.markdown("---")
st.caption("📡 AIDI Gossip Protocol | Controlo Descentralizado")
