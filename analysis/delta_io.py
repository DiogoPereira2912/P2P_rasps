"""
Leitura de tabelas Delta Lake (as pastas em data_exports/) só com pyarrow, sem
precisar do pacote deltalake no Mac.

Uma tabela Delta = ficheiros .parquet + _delta_log/*.json. Cada commit do log diz
que ficheiros foram adicionados ("add") ou removidos ("remove"); os ficheiros válidos
são os adicionados e não removidos.
"""
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def active_files(table_path):
    """Lista os .parquet ativos da tabela, pela ordem dos commits."""
    table_path = Path(table_path)
    log_dir = table_path / "_delta_log"
    if not log_dir.exists():
        return []

    files = {}
    for commit in sorted(log_dir.glob("*.json")):
        for line in commit.read_text().splitlines():
            if not line.strip():
                continue
            action = json.loads(line)
            if "add" in action:
                files[action["add"]["path"]] = True
            elif "remove" in action:
                files.pop(action["remove"]["path"], None)
    return [table_path / f for f in files]


def read_delta(table_path):
    """
    Lê a tabela inteira para um DataFrame. Tabelas com evolução de schema
    (schema_mode="merge") ficam com NaN nas colunas que faltam nos ficheiros antigos.
    """
    frames = [pq.read_table(f).to_pandas() for f in active_files(table_path)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
