import pandas as pd

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