from collections import Counter

# def aggregate_avg(params_dict):
#     """
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

def federated_avg(self, models_state_dicts):
    """
    Algoritmo FedAvg: Recebe uma lista de pesos e devolve a média.
    """
    if not models_state_dicts:
        return None
    avg_weights = copy.deepcopy(models_state_dicts[0]) # estrutura base - 1o modelo
    for key in avg_weights.keys():
        if torch.is_floating_point(avg_weights[key]):
            
            tensors = []
            for m in models_state_dicts:
                if key in m:
                    tensors.append(m[key].float())
            
            if tensors:
                avg_weights[key] = torch.mean(torch.stack(tensors), dim=0)
    return avg_weights

ALGS_DICT = {
    "avg": federated_avg,
}
