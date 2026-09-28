from collections import Counter

# def aggregate_avg(params_dict):
#     """
#     Ideal para agregar hiperparâmetros numéricos contínuos. - ML tipo Random Forest, SVM, etc.
#     Aggregate hyperparameters by calculating the average value for each parameter
#     Args:
#         params_dict: Dict of dictionaries containing hyperparameters from different nodes
#     Returns:
#         aggregated_params: Dictionary with averaged hyperparameters
#     """
#     aggregated_params = {}
#     num_nodes = len(params_dict)
#     for _, node_params in params_dict.items():
#         for param, value in node_params.items():
#             if param not in aggregated_params:
#                 aggregated_params[param] = 0
#             aggregated_params[param] += value / num_nodes

#     return aggregated_params


# def aggregate_majority(params_dict):
#     """
#     Aggregate hyperparameters by selecting the majority value for each parameter
#     Args:
#         params_dict: Dict of dictionaries containing hyperparameters from different nodes
#     Returns:
#         aggregated_params: Dictionary with majority hyperparameters
#     """
#     aggregated_params = {}
#     for _, node_params in params_dict.items():
#         for param, value in node_params.items():
#             if param not in aggregated_params:
#                 aggregated_params[param] = []
#             aggregated_params[param].append(value)

#     for param, values in aggregated_params.items():
#         most_common_value, _ = Counter(values).most_common(1)[0]
#         aggregated_params[param] = most_common_value

#     return aggregated_params

# def federated_avg(self, models_state_dicts):
#     """
#     Ideal para trabalhar com modelos de Deep Learning.
#     Algoritmo FedAvg: Recebe uma lista de pesos e devolve a média.
#     Args: models_state_dicts = [model1.state_dict(), model2.state_dict(), ...]
#     """
#     if not models_state_dicts:
#         return None
#     avg_weights = copy.deepcopy(models_state_dicts[0]) # estrutura base - 1o modelo
#     for key in avg_weights.keys():
#         if torch.is_floating_point(avg_weights[key]):
            
#             tensors = []
#             for m in models_state_dicts:
#                 if key in m:
#                     tensors.append(m[key].float())
            
#             if tensors:
#                 avg_weights[key] = torch.mean(torch.stack(tensors), dim=0)
#     return avg_weights

# def aggregate_avg(probs_dict):
#     """
#     Args: probs_dict = {'node_1': [[0.1, 0.9]], 'node_2': [[0.2, 0.8]]}
#     Returns: [[0.15, 0.85]]
#     Ideal para agregar probabilidades de predição. Recebe um dicionário de probabilidades de predição 
#     e devolve a média para cada classe.
#     """
#     if not probs_dict: return None
    
#     first_key = list(probs_dict.keys())[0]
#     num_preds = len(probs_dict[first_key]) 
#     num_classes = len(probs_dict[first_key][0])
    
#     final_probs = []

#     for i in range(num_preds):
#         sum_probs = [0.0] * num_classes
#         count = 0
        
#         for node_id, preds_list in probs_dict.items():
#             if i < len(preds_list):
#                 for c in range(num_classes):
#                     sum_probs[c] += preds_list[i][c]
#                 count += 1
        
#         avg_probs = [p / count for p in sum_probs]
#         final_probs.append(avg_probs)
        
#     return final_probs

def aggregate_avg(probs_dict):
    """
    Args: probs_dict = {'node_1': [[0.1, 0.9]], 'node_2': [[0.2, 0.8]]}
    Returns: [[0.15, 0.85]]
    Ideal para agregar probabilidades de predição. Recebe um dicionário de probabilidades de predição e 
    devolve a média para cada classe. Protegido contra nós que enviam vetores de tamanhos diferentes.
    """
    if not probs_dict: 
        return None
    
    # Descobre o tamanho máximo do vetor de probabilidades entre todos os nós
    max_num_preds = 0
    max_num_classes = 0
    
    for node_id, preds_list in probs_dict.items():
        if len(preds_list) > max_num_preds:
            max_num_preds = len(preds_list)
        
        for p in preds_list:
            if len(p) > max_num_classes:
                max_num_classes = len(p)
                
    if max_num_classes == 0:
        return None

    final_probs = []

    # Iterar sobre cada predição de forma segura
    for i in range(max_num_preds):
        sum_probs = [0.0] * max_num_classes
        count = 0
        
        for node_id, preds_list in probs_dict.items():
            if i < len(preds_list):
                current_pred = preds_list[i]
                
                # Só soma até ao limite de classes que este nó conhece
                for c in range(min(max_num_classes, len(current_pred))):
                    sum_probs[c] += current_pred[c]
                count += 1
        
        if count > 0:
            avg_probs = [p / count for p in sum_probs]
            final_probs.append(avg_probs)
        else:
            final_probs.append([0.0] * max_num_classes)
        
    return final_probs

ALGS_DICT = {
    "avg": aggregate_avg,
}
