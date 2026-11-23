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