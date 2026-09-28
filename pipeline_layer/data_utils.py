from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from deltalake import DeltaTable
import yaml

def load_data():
    df = DeltaTable("data_exports/local_output_rasp2/").to_pandas()
    return df

def data_split(df, label_map, test_size=0.2, random_state=42):
    '''
    Divide o DataFrame em conjuntos de treino e teste
    '''
    df['GLOBAL_STATUS'] = df['GLOBAL_STATUS'].astype(str).str.strip()

    df['target_id'] = df['GLOBAL_STATUS'].map(label_map)
    df = df.dropna(subset=['target_id'])
    df['target_id'] = df['target_id'].astype(int)

    counts = df['target_id'].value_counts()
    classes_validas = counts[counts >= 5].index
    df = df[df['target_id'].isin(classes_validas)]
    
    print("\n--- DEBUG: Classes que vão para o treino (>= 5 amostras) ---")
    print(df['target_id'].value_counts())

    X = df.drop(columns=['GLOBAL_STATUS', 'target_id', 'ts']) 
    y = df['target_id']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        shuffle=True,
        stratify=y # Garante que a proporção de classes se mantém
    )
    
    return X_train, X_test, y_train, y_test

# def build_param_grid(param_grid):
#     ''' 
#     Formata o dicionário de hyperparâmetros para o Pipeline Sklearn
#     Args:
#         param_grid: Dicionário com os hyperparâmetros
#     Returns:
#         formatted_grid: Dicionário formatado para o Pipeline Sklearn
#     '''
#     step_name = "classifier__"
#     formatted_grid = {}
#     for key, value in param_grid.items():
#         formatted_grid[f"{step_name}{key}"] = value
#     return formatted_grid

def resolve_targets_by_index(known_peers, indices_list):
    """
    Recebe a lista crua de peers descobertos e a lista de indices para routing.
    Retorna os IPs dos alvos.
    """
    targets = []
    for idx in indices_list:
        for peer in known_peers:
            if idx == peer[1]:
                target_ip = peer[0]
                targets.append(target_ip)
    print("Targets resolved:", targets)
    return targets

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