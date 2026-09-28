import yaml

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

# Teste
# if __name__ == "__main__":
#     l_map, r_map, c_list = load_class_mappings()
#     print("Mapa de Treino:", l_map)
#     print("Mapa Reverso:", r_map)
#     print("Classes por ordem:", c_list)