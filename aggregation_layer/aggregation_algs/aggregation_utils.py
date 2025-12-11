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