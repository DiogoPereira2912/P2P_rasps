import pandas as pd
import yaml

def process_df(df):
    df.sort_values("ts", inplace=True)
    df['ts'] = pd.to_datetime(df['ts'], unit='s')
    df['keypoints'] = df['keypoints'].tolist()
    return df

def process_labelled_df(df):
    # ISO8601: aceita timestamps com e sem fração de segundo na mesma coluna
    # (ex: '14:30:00' e '14:30:00.200000'), que chegam assim via JSON dos peers.
    # Ordena DEPOIS de converter (o merge_asof exige ts ordenado como data, não como texto)
    df['ts'] = pd.to_datetime(df['ts'], format="ISO8601")
    df.sort_values("ts", inplace=True)
    return df

def merge_labelled_dfs(dfs:list):
    """
    Mescla múltiplos dataframes rotulados de diferentes peers.
    Args:
        dfs (list): Lista de dataframes rotulados.
    Returns:
        pd.DataFrame: Dataframe mesclado.
    """
    if not dfs:
        return pd.DataFrame()  # Retorna um DataFrame vazio se a lista estiver vazia

    result = dfs[0].copy()
    for df in dfs[1:]:
        result = pd.merge_asof(result, df, on='ts', direction='nearest', tolerance=pd.Timedelta("500ms"))
        result.dropna(inplace=True)
    result.sort_values("ts", inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result

def status_from_votes(votes, rules_config, min_nos_override=None):
    """
    Critério ÚNICO do estado do shop floor, usado pelo labeler (ground truth) e pelo
    aggregation (consenso), para os dois serem diretamente comparáveis.

    Args:
        votes: {label: nº de nós que "veem" essa classe}
               - labeler:     nós com a coluna <label>_<nó> = 1
               - aggregation: nós com probabilidade >= prob_threshold
        rules_config: conteúdo do global_rules.yaml
        min_nos_override: força o nº mínimo de nós (ex: 1 para o label LOCAL de treino)
    Returns:
        label da regra disparada com maior prioridade (nº mais baixo), ou "SAFE"
    """
    default_min = rules_config.get("min_nos_default", 1)
    fired = []
    for rule in rules_config.get("regras", []):
        needed = min_nos_override or rule.get("min_nos", default_min)
        if votes.get(rule["label"], 0) >= needed:
            fired.append(rule)
    if not fired:
        return "SAFE"
    return min(fired, key=lambda r: r["prioridade"])["label"]


def load_class_mappings(yaml_path="labeling/global_rules.yaml"):
    """
    Lê o ficheiro de regras e gera os mapas de classes baseados na prioridade.
    Garante que a classe 0 é sempre 'SAFE'.
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    
    rules = data.get("regras", [])
    sorted_rules = sorted(rules, key=lambda x: x['prioridade'])
    class_list = ["SAFE"] + [r['label'] for r in sorted_rules]
    label_map = {name: i for i, name in enumerate(class_list)}
    reverse_label_map = {i: name for i, name in enumerate(class_list)}
    return label_map, reverse_label_map, class_list

# ---------------------------------------------------------------------------
# Regras de pose v2 (labeling/rules_v2.yaml): independentes da distância à câmara
# ---------------------------------------------------------------------------
import math

NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 0, 5, 6, 7, 8, 9, 10, 11, 12
L_ANK, R_ANK = 15, 16
BODY_RULE_KEYS = ("torso_angle_min", "bbox_aspect_min", "wrists_above_nose", "arm_extended",
                  "hip_to_ankle_max")


def _kp(kpts, i, min_conf):
    """(x, y) do keypoint i, ou None se não existir ou tiver confiança baixa."""
    try:
        x, y, c = kpts[i][:3]
    except (IndexError, TypeError, ValueError):
        return None
    return (x, y) if c >= min_conf and x >= 0 and y >= 0 else None


def _mid(a, b):
    if a and b:
        return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    return a or b


def _angle_at(a, b, c):
    """Ângulo (graus) no ponto b, entre os segmentos b->a e b->c."""
    v1, v2 = (a[0] - b[0], a[1] - b[1]), (c[0] - b[0], c[1] - b[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return None
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos))


def body_conditions_ok(kpts, bbox, cond, min_conf=0.3):
    """
    Verifica as condições de pose v2 de uma regra para UMA pessoa.
    kpts: 17 keypoints [x, y, conf] (coordenadas da imagem; y cresce para baixo)
    bbox: (x1, y1, x2, y2)
    Devolve False se faltar algum keypoint necessário.
    """
    shoulder = _mid(_kp(kpts, L_SH, min_conf), _kp(kpts, R_SH, min_conf))
    hip = _mid(_kp(kpts, L_HIP, min_conf), _kp(kpts, R_HIP, min_conf))
    if not (shoulder and hip):
        return False
    torso = math.hypot(shoulder[0] - hip[0], shoulder[1] - hip[1])
    if torso == 0:
        return False

    if "torso_angle_min" in cond:
        # 0° = tronco vertical (de pé), 90° = horizontal (deitado)
        angle = math.degrees(math.atan2(abs(shoulder[0] - hip[0]), abs(shoulder[1] - hip[1])))
        if angle < cond["torso_angle_min"]:
            return False

    if "bbox_aspect_min" in cond:
        x1, y1, x2, y2 = bbox
        if y2 - y1 <= 0 or (x2 - x1) / (y2 - y1) < cond["bbox_aspect_min"]:
            return False

    if "hip_to_ankle_max" in cond:
        # Agachado / no chão: a anca desce até perto dos pés. Distância VERTICAL anca->tornozelo
        # em comprimentos de tronco (de pé ~1.0; de cócoras ~0.4). Precisa dos pés visíveis.
        ankle = _mid(_kp(kpts, L_ANK, min_conf), _kp(kpts, R_ANK, min_conf))
        if not ankle or abs(ankle[1] - hip[1]) / torso > cond["hip_to_ankle_max"]:
            return False

    if "wrists_above_nose" in cond:
        nose = _kp(kpts, NOSE, min_conf)
        wrists = [_kp(kpts, L_WR, min_conf), _kp(kpts, R_WR, min_conf)]
        if not nose or not all(wrists):
            return False
        margin = cond["wrists_above_nose"] * torso
        if not all(w[1] < nose[1] - margin for w in wrists):
            return False

    if "arm_extended" in cond:
        p = cond["arm_extended"]
        extended = False
        for sh_i, el_i, wr_i in ((L_SH, L_EL, L_WR), (R_SH, R_EL, R_WR)):
            sh, el, wr = (_kp(kpts, i, min_conf) for i in (sh_i, el_i, wr_i))
            if not (sh and el and wr):
                continue
            elbow = _angle_at(sh, el, wr)
            reach = abs(wr[0] - sh[0]) / torso  # componente horizontal: exclui braço para cima/baixo
            if elbow is not None and elbow >= p.get("elbow_angle_min", 150) and reach >= p.get("reach_min", 0.8):
                extended = True
        if not extended:
            return False

    return True


def frame_level(df_labelled):
    """
    Várias pessoas no mesmo frame -> uma linha por frame (ts): uma regra fica a 1 se
    PELO MENOS UMA pessoa a cumprir. Evita que o merge entre nós escolha uma pessoa ao acaso.
    """
    return df_labelled.groupby("ts", as_index=False).max()
