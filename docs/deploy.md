# Deploy para os Raspberries

Tudo corre **no Mac**, ligado à rede de desenvolvimento. Os Pis precisam apenas de Docker, do utilizador `aidi` e da tua chave SSH autorizada.

Há três coisas diferentes para enviar, com frequências muito diferentes:

| O que mudou | Comando | Quanto demora |
|---|---|---|
| Código Python, regras YAML, compose | `tools/deploy.sh` | segundos |
| IP / node_id / topologia de um nó | editar `inventory.yaml` → `tools/deploy.sh` | segundos |
| Dependências (`requirements.txt`, apt) | `control/images.sh build` → UI *Deploy Images* | minutos (imagem de ~5 GB) |

> 💡 O código **não** está dentro da imagem `p2p-rpi` — é montado por volumes no `compose.yaml`. Por isso, mudar código **nunca** obriga a recompilar a imagem.

---

## 1. Inventário — `inventory.yaml`

Único sítio onde estão os dados de cada Raspberry:

```yaml
nodes:
  rasp1:
    ip: 192.168.0.110
    node_id: 0
```

- **nome** (`rasp1`) → passa a ser o `device_id` (pastas em `data_exports/`, chaves nas regras).
- **ip** → `peer_ip`. Convém fixar o IP com uma reserva DHCP no router.
- **node_id** → índice usado nas topologias e no `central_id`.
- **defaults** → valores comuns a todos os nós (porta MQTT, modo, `min_peers`, câmara, ThingsBoard...).
- Qualquer valor pode ser substituído só para um nó, escrevendo-o dentro desse nó (ex: `routing_topology`).

**Topologia:** `null` = enviar para todos os **outros** nós (o próprio fica de fora para evitar duplicados).

**Tokens (ThingsBoard):** ficam em `inventory.secrets.yaml`, que é local e não vai para o git. Tem a mesma estrutura e é fundido por cima do inventário:

```yaml
defaults:
  tb_token: "..."
nodes:
  rasp2:
    tb_token: "token específico do rasp2"
```

### Adicionar / trocar um Raspberry
1. Acrescentar ou editar a entrada em `nodes:` (IP e `node_id` únicos — o script valida).
2. `tools/deploy.sh <nome>` — envia código + config só para esse nó.
3. A UI e o Prometheus atualizam-se sozinhos (ver [control.md](control.md); o Prometheus precisa de restart).

---

## 2. Gerar as configs — `tools/gen_configs.py`

```bash
python3 tools/gen_configs.py
```

Normalmente não é preciso correr à mão: o `deploy.sh` corre-o sempre primeiro. Gera:

| Ficheiro | Para quê |
|---|---|
| `build/nodes/<nó>/client/config.yaml` | config completa do nó |
| `build/nodes/<nó>/labeling/global_rules.yaml` | regras globais com o `device_id` do nó (a partir de `labeling/global_rules.example.yaml`) |
| `build/nodes.txt` | lista `nome ip user path` usada pelo `deploy.sh` |
| `control/prometheus.yaml` | alvos do Prometheus |

A pasta `build/` é recriada sempre — **nunca editar à mão**. Para mudar algo, editar `inventory.yaml` (ou o template das regras) e voltar a gerar.

---

## 3. Enviar código + config — `tools/deploy.sh`

Pela UI: tab **Setup & Deploy → 📤 Atualizar Código** (escolher nós + opção de reiniciar). Ou no terminal:

```bash
tools/deploy.sh                    # todos os nós
tools/deploy.sh rasp2 rasp4        # só alguns
tools/deploy.sh --restart          # e reinicia os serviços Python no fim
tools/deploy.sh --dry-run rasp1    # mostra o que seria enviado, sem alterar nada
```

Para cada nó:
1. `rsync` do código para `/home/aidi/P2P/` (respeita o `.rsyncignore`).
2. Copia `build/nodes/<nó>/` por cima (config e regras globais desse nó).
3. Com `--restart`: `docker compose restart peer pipeline aggregation labeler detector`.

**O que NUNCA é enviado nem apagado nos Pis** (`.rsyncignore`): `data_exports/`, `models/`, `results/`, `control/`, `roi_selection/`, `tools/`, `docs/`, `build/`, `inventory*.yaml`, `*.tar`.

> ⚠️ `video_server/rois_url*.yaml` **é** enviado — um deploy substitui as ROIs que estão no Pi pelas que estão no Mac. Se isso deixar de ser desejado, acrescentar `video_server/rois_url*.yaml` ao `.rsyncignore`.

> O `update_git.sh` antigo continua a funcionar, mas não envia as configs geradas. A ideia é substituí-lo pelo `deploy.sh` depois de testado.

---

## 4. Imagens Docker (deploy offline)

Os Pis não precisam de internet: as imagens vão do Mac como ficheiros `.tar`, guardados em `control/imgs/` (fora do git).

### Criar os `.tar` — `control/images.sh`

```bash
cd control

# imagem base do projeto (Python, Hailo, torch, ...) — só quando mudam dependências
./images.sh build p2p_img/base p2p-rpi            # -> imgs/p2p-rpi.tar

# imagens públicas usadas no compose.yaml (versão TEM de coincidir com o compose)
./images.sh pull eclipse-mosquitto:2.0 mosquitto              # -> imgs/mosquitto.tar
./images.sh pull gcr.io/cadvisor/cadvisor:v0.47.0 cadvisor    # -> imgs/cadvisor.tar (v0.49.x incompatível com o Docker dos Pis)
```

Tudo é compilado/descarregado para `linux/arm64` (Raspberry Pi 5).

### Enviar para os Pis
Pela UI: tab **Setup & Deploy → Deploy Images** → escolher nós e `.tar`.
Ou no terminal: `control/node.sh load <ip> control/imgs/p2p-rpi.tar`

Isto envia o `.tar`, faz `docker load` no Pi e apaga o `.tar` lá.

---

## 5. Operações num nó — `control/node.sh`

| Comando | O que faz | Botão na UI |
|---|---|---|
| `control/node.sh start <ip>` | `docker compose down && up -d` no Pi | 🔌 Levantar Serviço |
| `control/node.sh load <ip> <ficheiro.tar>` | envia e carrega uma imagem | 🚀 Deploy Images |

Utilizador e pasta remota: `aidi` e `/home/aidi/P2P` por omissão (alteráveis com `RASP_USER` / `RASP_DEST_DIR`).

### Comandos úteis diretamente no Pi

```bash
ssh aidi@192.168.0.110
cd ~/P2P
docker compose ps                      # estado dos serviços
docker compose logs -f pipeline        # logs de um serviço (peer, labeler, aggregation, detector, ...)
docker compose restart aggregation     # reiniciar só um serviço
cat mosquitto/conf.d/bridges.conf      # bridges criadas para os outros nós
ls data_exports/ models/               # dados e modelos guardados no nó
```

---

## 6. Trazer resultados para o Mac — `tools/fetch_results.sh`

```bash
tools/fetch_results.sh gossip_5nos              # todos os nós
tools/fetch_results.sh fl_central_rasp1 rasp1   # só alguns
```

Copia `data_exports/` e `results/` de cada Pi para:

```
experiments/2026-11-23_1430_gossip_5nos/
├── metadata.yaml      ← commit do git, se havia alterações por commitar, data
├── inventory.yaml     ← cópia do inventário usado
├── rasp1/data_exports/  rasp1/results/
└── ...
```

- Nunca apaga nada nos Pis (sem `--delete`).
- `experiments/` não vai para o git nem para os Pis.
- Usar um nome por experiência (ex: `gl_5nos_rep1`, `fl_5nos_rep1`) para depois comparar.

---

## 7. Sincronizar relógios — `tools/sync_clocks.py`

A rede de desenvolvimento não tem internet, por isso os Pis não têm NTP e os relógios
desviam-se (já se mediram **até 27 dias** de diferença entre nós). Os dados dos vários nós
são juntados pelo **timestamp real** (tolerância de 500 ms entre nós), por isso os relógios
têm de estar acertados **antes de cada experiência**.

```bash
python3 tools/sync_clocks.py --check   # só mede o desfasamento de cada Pi
python3 tools/sync_clocks.py           # acerta todos os Pis pelo relógio do Mac
python3 tools/sync_clocks.py rasp3     # só um nó
```

```
nó         offset antes  offset depois      rtt
rasp1        +27.0 dias        +2.1 ms    8.5ms ✅
```

- ✅ = desfasamento final ≤ 50 ms. Guardar esta tabela: é útil para reportar no artigo.
- **Método** usado em cada nó:
  - `sudo`: `sudo -n date -s` (se o utilizador `aidi` tiver sudo sem password).
  - `docker`: se o sudo pedir password, corre `date -s` num contentor `p2p-rpi` com
    `--cap-add SYS_TIME` (o `aidi` já está no grupo docker). A correção é relativa e
    calculada no Pi, por isso o arranque do contentor não afeta a precisão.
  - Para usar sempre `sudo`, uma vez por Pi (pede a password):
    `ssh -t aidi@<ip> "echo 'aidi ALL=(root) NOPASSWD: /usr/bin/date' | sudo tee /etc/sudoers.d/gossip-date && sudo chmod 440 /etc/sudoers.d/gossip-date"`
- Sintoma de relógios dessincronizados no labeler:
  `⚠️ Merge vazio: nenhuma linha alinhada entre nós`.
- Os relógios voltam a desviar-se com o tempo (sem NTP): correr antes de cada sessão de testes.

---

## Primeira instalação de um Pi novo

1. Docker instalado e utilizador `aidi` no grupo `docker`.
2. Chave SSH do Mac autorizada: `ssh-copy-id aidi@<ip>`.
3. Adicionar o Pi ao `inventory.yaml`.
4. Carregar as imagens (`p2p-rpi`, `mosquitto`, `cadvisor`) — secção 4.
5. `tools/deploy.sh <nome>`
6. UI → 🔌 Levantar Serviço.
