from collections import Counter
import numpy as np
from labeling.labeling_utils import status_from_votes  # mesmo critério do ground truth

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

def confidence_fallback_aggregation(params_dict, threshold=0.3, fallback_mode="priority", local_id=None, priority_map=None):
    """
    Agrega probabilidades e aplica fallback se a margem de confiança for baixa.
    No modo 'priority', escolhe a opção com MAIOR prioridade (número mais baixo) entre as duas mais votadas.
    """
    # Só a 1ª predição de cada nó entra na decisão (flat_probs = avg_probs[0]).
    # Cortar já aqui evita o erro "inhomogeneous shape" quando os nós enviam batches
    # com nº de linhas diferente.
    probs_list = [np.asarray(p, dtype=float)[:1] for p in params_dict.values()]

    num_classes = {p.shape[1] for p in probs_list}
    if len(num_classes) > 1:
        raise ValueError(
            f"Nós com nº de classes diferente {num_classes}: confirmar que todos usam o mesmo global_rules.yaml"
        )

    avg_probs = np.mean(probs_list, axis=0)
    flat_probs = avg_probs[0] 
    
    sorted_indices = np.argsort(flat_probs)[::-1]
    top1_idx = sorted_indices[0]
    top2_idx = sorted_indices[1]
    
    margin = flat_probs[top1_idx] - flat_probs[top2_idx]
    
    if margin >= threshold:
        print(f"✅ Confiança Alta (Margem: {margin:.2f}). Decisão Global mantida.")
        return avg_probs
    else:
        print(f"⚠️ Confiança Baixa (Margem: {margin:.2f} < {threshold}). A aplicar Fallback...")
        
        if fallback_mode == "priority" and priority_map is not None:
            # Consulta a prioridade (se não existir, assume 999 = muito baixa prioridade / SAFE)
            prio_1 = priority_map.get(top1_idx, 999)
            prio_2 = priority_map.get(top2_idx, 999)
            
            # Escolhe o índice com MENOR número (MAIOR prioridade)
            preventive_idx = top1_idx if prio_1 <= prio_2 else top2_idx
            
            print(f"🛡️ Conflito: idx {top1_idx} (prio {prio_1}) vs idx {top2_idx} (prio {prio_2}). Venceu o idx {preventive_idx}.")
            
            # Devolve a predição com 100% de probabilidade na classe vencedora
            fallback_probs = np.zeros_like(avg_probs)
            fallback_probs[0, preventive_idx] = 1.0
            return fallback_probs
            
        elif fallback_mode == "local" and local_id in params_dict:
            return params_dict[local_id]
        
        return avg_probs

def rules_aggregation(params_dict, rules_config=None, label_map=None, verbose=True, **kwargs):
    """
    Consenso com o MESMO critério do ground truth (global_rules.yaml):
      1. um nó "vê" a classe c se a sua probabilidade em c >= prob_threshold
      2. a regra dispara se >= min_nos nós a virem
      3. entre as regras disparadas vence a de maior prioridade; nenhuma -> SAFE
    Usa a 1ª predição de cada nó (igual aos outros métodos).
    Nota: se houver menos nós na ronda do que o min_nos de uma regra, ela não pode disparar.

    Returns:
        vetor one-hot (1, n_classes) com a classe decidida
    """
    probs = np.array([np.asarray(p, dtype=float)[0] for p in params_dict.values()])
    threshold = rules_config.get("prob_threshold", 0.5)

    votes = {
        label: int((probs[:, idx] >= threshold).sum())
        for label, idx in label_map.items()
        if label != "SAFE"
    }
    decision = status_from_votes(votes, rules_config)
    fired = {k: v for k, v in votes.items() if v > 0}
    if verbose:
        print(f"🗳️  Votos por classe ({len(params_dict)} nós): {fired or 'nenhum'} -> {decision}")

    result = np.zeros((1, probs.shape[1]))
    result[0, label_map[decision]] = 1.0
    return result


ALGS_DICT = {
    "avg": aggregate_avg,
    "fallback": confidence_fallback_aggregation,
    "rules": rules_aggregation,
}
