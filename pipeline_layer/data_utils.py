from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits

def load_data():
    df = load_wine(as_frame=True).frame
    return df

def data_split(df, test_size=0.2, random_state=42):
    '''
    Divide o DataFrame em conjuntos de treino e teste
    Args:
        df: DataFrame contendo os dados
        test_size: Proporção dos dados a serem usados para teste
        random_state: Semente para reprodução dos resultados
    Returns:
        X_train, X_test, y_train, y_test: Dados divididos em treino e teste
    '''
    X = df.iloc[:, :-1]  # Assuming all columns except the last are features
    y = df.iloc[:, -1]  # Assuming the target variable is the last column
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def build_param_grid(param_grid):
    ''' 
    Formata o dicionário de hyperparâmetros para o Pipeline Sklearn
    Args:
        param_grid: Dicionário com os hyperparâmetros
    Returns:
        formatted_grid: Dicionário formatado para o Pipeline Sklearn
    '''
    step_name = "classifier__"
    formatted_grid = {}
    for key, value in param_grid.items():
        formatted_grid[f"{step_name}{key}"] = value
    return formatted_grid

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