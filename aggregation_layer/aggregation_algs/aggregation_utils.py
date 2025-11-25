def resolve_targets_by_index(known_peers, indices_list):
    """
    Recebe a lista crua de peers descobertos e a lista de indices para routing.
    Retorna os IPs dos alvos.
    """
    sorted_peers = sorted(known_peers)
    
    targets = []
    for idx in indices_list:
        if idx < len(sorted_peers):
            target_ip = sorted_peers[idx]
            targets.append(target_ip)
        else:
            print(f"[ROUTING] Indice {idx} pedido, mas só conheço {len(sorted_peers)} peers.")
    print("Targets resolved:", targets)
    return targets