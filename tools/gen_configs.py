#!/usr/bin/env python3
"""
Gera os ficheiros específicos de cada nó a partir de inventory.yaml
(build/nodes/<nó>/... e control/prometheus.yaml).
Modo de uso: docs/deploy.md (secção "Gerar as configs")
"""
import copy
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
INVENTORY = ROOT / "inventory.yaml"
SECRETS = ROOT / "inventory.secrets.yaml"
GLOBAL_RULES_TEMPLATE = ROOT / "labeling" / "global_rules.example.yaml"
BUILD = ROOT / "build"
PROMETHEUS = ROOT / "control" / "prometheus.yaml"

HEADER = (
    "# ============================================================\n"
    "# GERADO AUTOMATICAMENTE por tools/gen_configs.py — NÃO EDITAR.\n"
    "# Altera inventory.yaml e volta a correr o script.\n"
    "# ============================================================\n"
)

# Campos que não pertencem ao client/config.yaml
NODE_META_KEYS = {"ip"}


def load_inventory():
    with open(INVENTORY) as f:
        inv = yaml.safe_load(f)
    # Segredos (tokens ThingsBoard, ...) num ficheiro local fora do git, mesma estrutura
    if SECRETS.exists():
        with open(SECRETS) as f:
            inv = deep_merge(inv, yaml.safe_load(f) or {})
        print(f"🔑 Segredos carregados de {SECRETS.name}")
    else:
        print(f"⚠️  {SECRETS.name} não encontrado -> tokens ficam vazios")
    for key in ("ssh_user", "remote_path", "defaults", "nodes"):
        if key not in inv:
            sys.exit(f"❌ inventory.yaml: falta a chave '{key}'")
    return inv


def validate(nodes):
    ids, ips = {}, {}
    for name, n in nodes.items():
        for key in ("ip", "node_id"):
            if key not in n:
                sys.exit(f"❌ Nó '{name}': falta '{key}'")
        if n["node_id"] in ids:
            sys.exit(f"❌ node_id {n['node_id']} repetido em '{name}' e '{ids[n['node_id']]}'")
        if n["ip"] in ips:
            sys.exit(f"❌ IP {n['ip']} repetido em '{name}' e '{ips[n['ip']]}'")
        ids[n["node_id"]] = name
        ips[n["ip"]] = name


def deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def build_node_config(name, node, defaults, all_ids):
    cfg = deep_merge(defaults, {k: v for k, v in node.items() if k not in NODE_META_KEYS})
    cfg["peer_ip"] = node["ip"]
    cfg["device_id"] = name

    # null na topologia = todos os OUTROS nós (o próprio fica de fora para evitar duplicados)
    topo = cfg.setdefault("routing_topology", {})
    for key in ("aggregation_topology", "pipeline_topology"):
        if topo.get(key) is None:
            topo[key] = [i for i in all_ids if i != node["node_id"]]
    return cfg


def build_global_rules(device_id):
    """Copia o template e troca a chave do rasp nas condições pelo device_id do nó."""
    with open(GLOBAL_RULES_TEMPLATE) as f:
        rules = yaml.safe_load(f)
    for rule in rules.get("regras", []):
        cond = rule.get("condicoes", {})
        rule["condicoes"] = {device_id: v for v in cond.values()} if cond else {}
    return rules


def write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(HEADER)
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def build_prometheus(ips):
    return {
        "global": {"scrape_interval": "2s", "evaluation_interval": "2s"},
        "scrape_configs": [
            {"job_name": "prometheus", "static_configs": [{"targets": ["localhost:9090"]}]},
            {"job_name": "hardware_cadvisor", "static_configs": [{"targets": [f"{ip}:8080" for ip in ips]}]},
            {"job_name": "throughput_python", "static_configs": [{"targets": [f"{ip}:8000" for ip in ips]}]},
        ],
    }


def main():
    inv = load_inventory()
    nodes = inv["nodes"]
    validate(nodes)

    if BUILD.exists():
        shutil.rmtree(BUILD)

    all_ids = sorted(n["node_id"] for n in nodes.values())
    ips = [n["ip"] for n in nodes.values()]

    for name, node in nodes.items():
        node_dir = BUILD / "nodes" / name
        write_yaml(node_dir / "client" / "config.yaml",
                   build_node_config(name, node, inv["defaults"], all_ids))
        write_yaml(node_dir / "labeling" / "global_rules.yaml", build_global_rules(name))
        print(f"✅ {name:8s} {node['ip']:15s} node_id={node['node_id']}")

    with open(BUILD / "nodes.txt", "w") as f:
        for name, node in nodes.items():
            f.write(f"{name} {node['ip']} {inv['ssh_user']} {inv['remote_path']}\n")

    write_yaml(PROMETHEUS, build_prometheus(ips))
    print(f"✅ {PROMETHEUS.relative_to(ROOT)} atualizado")

    print(f"\n📁 Ficheiros gerados em {BUILD.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
