import pandas as pd
import yaml

def process_df(df):
    df.sort_values("ts", inplace=True)
    df['ts'] = pd.to_datetime(df['ts'], unit='s')
    df['keypoints'] = df['keypoints'].tolist()
    return df

def process_labelled_df(df):
    df.sort_values("ts", inplace=True)
    df['ts'] = pd.to_datetime(df['ts'])
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