# Guião experimental — Shop floor simulado

Objetivo: gerar dados que permitam responder, com rigor estatístico, a:

- **Q1** — O consenso entre nós (GL / FL) aproxima-se mais do estado real do shop floor do que cada nó sozinho?
- **Q2** — GL e FL têm a mesma qualidade de decisão? Que custo tem cada um (latência, tráfego, recursos)?
- **Q3** — O que acontece quando um nó falha (em FL: o servidor central)?

O cenário tem eventos **encenados com tempos conhecidos**. Isto dá, além do ground truth das
regras, um registo independente do que realmente aconteceu (ver §6) — responde à crítica
"o ground truth vem das mesmas regras que geram as features".

---

## 0. Antes de tudo: calibração das regras (uma vez, ~30 min)

As regras locais (`labeling/rules.yaml`) são o que transforma poses em classes. Ao rever o
código encontrámos comportamentos que **não correspondem à intenção** descrita nos comentários
— é preciso medi-los antes de encenar o que quer que seja:

| Regra | O que o código faz realmente | Efeito |
|---|---|---|
| Todas as `pose_rule` | distância **em píxeis** entre 2 keypoints ≤ `max_dist` | Depende da distância da pessoa à câmara: longe da câmara o corpo é pequeno e quase todas as distâncias ficam abaixo do limiar |
| `PERIGO_GRUA` | pulsos a ≤ 50 px | Pessoa longe da câmara com as mãos em baixo pode disparar (pulsos próximos em píxeis) — **provável causa do excesso de GRUA nos dados de hoje** |
| `PROXIMIDADE_MAQUINA` | pulso–ombro ≤ 150 px → **braço dobrado** | O comentário diz "braço esticado = possível toque", mas dispara com o braço encolhido |
| `SINAL_SOS` | pulso direito–nariz ≤ 100 px | Mão junto à cara (ex: telefone, coçar), não "mãos acima da cabeça" |
| `QUEDA_DETETADA` | nariz–anca ≤ 30 px (distância euclidiana) | Cabeça junto à anca na imagem (pessoa dobrada/enrolada), não necessariamente deitada |
| `INTRUSAO` | centro da pessoa dentro de (10..1900, 10..1000) | Praticamente qualquer pessoa visível → "INTRUSAO" = **presença** |
| `ZONA_FORKLIFT` | centro em x ∈ [1500, 1920] | As coordenadas das deteções são relativas ao **recorte da ROI**, não à imagem completa: se o recorte for mais estreito que 1500 px, **nunca dispara** |

**Procedimento:** para cada câmara e cada ação da §2, uma pessoa faz a ação 10 s em 3 distâncias
(perto / meio / longe). Registar que colunas `<CLASSE>_<nó>` ficaram a 1 (logs do labeler ou
`raw_<nó>`). Resultado: tabela *ação × câmara × distância → classe disparada*.

Se a tabela mostrar disparos errados (muito provável para GRUA/PROXIMIDADE longe da câmara),
**corrigir as regras antes de recolher dados para o artigo**. Sugestões (a decidir):
- normalizar as distâncias pelo tamanho do corpo (ex: dividir pela largura dos ombros ou pela
  altura da bounding box) — deixa de depender da distância à câmara;
- `ROI_rule` em coordenadas do recorte, ou somar a origem do recorte às coordenadas;
- rever a intenção de PROXIMIDADE e SOS.

> As regras mudam o ground truth → **não mudar nada nas regras entre runs que vão ser comparados.**

---

## 0b. Regras v2 (em uso)

As regras acima (v1, `labeling/rules.yaml` + `global_rules.example.yaml`) foram substituídas por
um conjunto mais pequeno e independente da distância à câmara. **As v1 continuam no repo**; a
versão em uso escolhe-se no `inventory.yaml` (`local_rules` e `global_rules_template`).

| Prioridade | Classe | Condição (por pessoa) | `min_nos` |
|---|---|---|---|
| 1 | `QUEDA` | tronco a ≥ 60° da vertical **e** bounding box mais larga do que alta (≥ 1.0) | 1 |
| 2 | `SINAL_SOS` | os **dois** pulsos acima do nariz (margem 0.1 × tronco) | 2 |
| 3 | `BRACO_ESTENDIDO` | cotovelo ≥ 150° **e** pulso afastado do ombro na horizontal ≥ 0.8 × tronco | 2 |
| — | `SAFE` | nenhuma regra (pessoas a trabalhar normalmente) | — |

- Ficheiros: `labeling/rules_v2.yaml` (locais) e `labeling/global_rules_v2.example.yaml` (globais).
- Várias pessoas no mesmo frame: a câmara dispara a regra se **pelo menos uma** pessoa a cumprir.
- Teste com poses sintéticas (perto/longe da câmara, gestos e armadilhas):
  `python3 labeling/test_rules_v2.py`. Correr outra vez depois de mudar limiares.
- **Limiar a confirmar na calibração:** `bbox_aspect_min` da QUEDA. Baixar-se para apanhar uma
  caixa dá ≈ 0.9 na pose sintética (logo abaixo de 1.0); se der falsas quedas, subir para ~1.3.
- O `BRACO_ESTENDIDO` usa a componente **horizontal** do braço: um braço estendido na direção da
  câmara fica encurtado na imagem e pode não disparar nessa câmara — outra câmara com outro
  ângulo vê-o (bom exemplo da colaboração entre câmaras para o artigo).

**Versão de simulação** (`labeling/rules_v2_sim.yaml`, a usada por defeito no `inventory.yaml`):
igual à v2, mas a **QUEDA é agachar-se até ao chão** (de cócoras) em vez de deitar — mais fácil de
encenar no escritório. Deteção: distância vertical anca→tornozelo ≤ 0.5 × tronco. **Os pés têm de
aparecer na imagem.** No artigo, descrever como "gesto substituto para simular uma queda".
Teste: `python3 labeling/test_rules_v2.py labeling/rules_v2_sim.yaml`.

**Armadilhas a encenar** (não devem disparar): baixar-se para apanhar uma caixa, levantar só um
braço, apontar com o braço dobrado.

## 1. Mapa de câmaras e zonas (preencher)

Desenhar a planta do espaço e marcar que câmaras veem cada zona. É isto que decide se um
evento é visto por 1 câmara (testa `min_nos: 1`) ou por várias (testa a confirmação, `min_nos: 2`).

| Zona | Descrição | rasp1 | rasp2 | rasp3 | rasp4 | rasp5 | Nº câmaras |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|
| A | Grua (sobreposição) | ✔ | ✔ | | | | 2 |
| B | Máquina (sobreposição) | | ✔ | ✔ | | | 2 |
| C | Corredor logístico (sobreposição) | | | | ✔ | ✔ | 2 |
| D | Bancada isolada | | | ✔ | | | 1 |
| E | Armazém isolado | | | | | ✔ | 1 |

*(exemplo — substituir pelo mapa real; guardar uma foto/planta para a figura do setup no artigo)*

**Requisito:** pelo menos uma zona vista por 1 só câmara e duas vistas por ≥2 câmaras.

---

## 2. Dicionário de ações

Cada ação tem de disparar **a regra pretendida** na calibração (§0). Se não disparar, ajustar a
forma de encenar (ou a regra) antes de continuar.

| Código | Classe pretendida | Como encenar | Duração |
|---|---|---|---|
| `VAZIO` | SAFE | ninguém no campo de visão | 20 s |
| `PRESENCA` | INTRUSAO (com as regras atuais) | caminhar normalmente, sem gestos | 15 s |
| `GRUA` | PERIGO_GRUA | de frente para a câmara, pulsos juntos à frente do peito (sinal de comando de grua) | 15 s |
| `QUEDA` | QUEDA_DETETADA | deitar no chão (usar tapete), de lado para a câmara | 15 s |
| `SOS` | SINAL_SOS | mão direita junto à cara / acima da cabeça (conforme a calibração) | 15 s |
| `CORREDOR` | ZONA_FORKLIFT | entrar e ficar no corredor logístico | 15 s |
| `MAQUINA` | PROXIMIDADE_MAQUINA | junto à "máquina", braço em interação (conforme a calibração) | 15 s |

Entre eventos: **10 s de saída** do campo de visão (VAZIO), para os eventos ficarem separados.

---

## 3. Cenário A — operação normal (~5 min)

Um ator. Cada classe de perigo aparece **2 vezes**: numa zona com 2 câmaras e numa zona com 1 câmara.
Ficheiro com os tempos: [`cenarios/cenario_A.csv`](cenarios/cenario_A.csv).

| Início | Fim | Ação | Zona | Câmaras | Esperado com `min_nos_default: 2` |
|---|---|---|---|---|---|
| 0:00 | 0:20 | VAZIO | — | — | SAFE |
| 0:20 | 0:35 | PRESENCA | A | 2 | INTRUSAO |
| 0:35 | 0:50 | GRUA | A | 2 | PERIGO_GRUA |
| 0:50 | 1:00 | VAZIO | — | — | SAFE |
| 1:00 | 1:15 | MAQUINA | B | 2 | PROXIMIDADE_MAQUINA |
| 1:15 | 1:25 | VAZIO | — | — | SAFE |
| 1:25 | 1:40 | SOS | B | 2 | SINAL_SOS |
| 1:40 | 1:50 | VAZIO | — | — | SAFE |
| 1:50 | 2:05 | QUEDA | B | 2 | QUEDA_DETETADA |
| 2:05 | 2:15 | VAZIO | — | — | SAFE |
| 2:15 | 2:30 | CORREDOR | C | 2 | ZONA_FORKLIFT |
| 2:30 | 2:40 | VAZIO | — | — | SAFE |
| 2:40 | 2:55 | GRUA | D | 1 | SAFE ¹ |
| 2:55 | 3:05 | VAZIO | — | — | SAFE |
| 3:05 | 3:20 | QUEDA | D | 1 | SAFE ¹ (ou QUEDA se `min_nos: 1`) |
| 3:20 | 3:30 | VAZIO | — | — | SAFE |
| 3:30 | 3:45 | SOS | E | 1 | SAFE ¹ |
| 3:45 | 3:55 | VAZIO | — | — | SAFE |
| 3:55 | 4:10 | MAQUINA | E | 1 | SAFE ¹ |
| 4:10 | 4:20 | VAZIO | — | — | SAFE |
| 4:20 | 4:35 | PRESENCA | D | 1 | SAFE ¹ |
| 4:35 | 5:00 | VAZIO | — | — | SAFE |

¹ Visto por 1 câmara só: com `min_nos: 2` o shop floor considera-o não confirmado. É **de propósito**:
mostra no artigo o efeito do `min_nos` (fig. 5 recalcula k = 1, 2, 3 sobre os mesmos dados).

**Variante com 2 atores (opcional, cenário A2):** repetir com dois eventos simultâneos em zonas
diferentes (ex: GRUA em A + QUEDA em B) para testar a prioridade (GRUA prioridade 1 vence).

---

## 4. Cenário B — falha de um nó (~5 min)

Igual ao cenário A, mas aos **2:00** desliga-se um nó durante 60 s e volta a ligar-se:

```bash
ssh aidi@<ip> "cd ~/P2P && docker compose stop pipeline aggregation"   # 2:00
ssh aidi@<ip> "cd ~/P2P && docker compose start pipeline aggregation"  # 3:00
```

| Run | Nó desligado | O que se espera |
|---|---|---|
| FL | o **servidor central** | sem decisões globais durante 60 s (ponto único de falha) |
| FL | um worker | o central continua, com menos 1 nó |
| GL | o mesmo nó que o central do FL | os restantes continuam a decidir |

Registar a hora exata de paragem/arranque (o registo de eventos da §6 faz isso).

---

## 5. Protocolo de execução

**Plano mínimo para o artigo:**

| Cenário | Modo | Repetições |
|---|---|---|
| A | GL | 5 |
| A | FL (central = rasp1) | 5 |
| B | GL | 3 |
| B | FL (falha do central) | 3 |

Ordem **alternada** (A-GL, A-FL, A-GL, A-FL, ...) para o cansaço do ator e a luz da sala não
favorecerem um dos modos.

**Checklist antes de cada sessão**
1. `python3 tools/sync_clocks.py` → ✅ em todos (guardar a tabela).
2. `tools/deploy.sh --restart` se houve mudanças de código. **Não mudar regras entre runs comparados.**
3. Modelos treinados presentes (`ls ~/P2P/models/` em cada Pi).
4. Grafana aberto (dashboards *Gossip*) para verificar que todos os nós estão UP.

**Em cada run**
1. UI → tab Inferência → modo GL ou FL → etiqueta: `cenarioA_rep1` (o run_id fica `<data>_<GL|FL>_cenarioA_rep1`).
2. "Inferências por batch" (barra lateral) suficiente para cobrir os 5 min. No teste de 23/11,
   10 batches ≈ 35 s → **≈ 90 batches para 5 min** (confirmar no primeiro run e ajustar).
3. Iniciar a inferência → aguardar 10 s → começar o guião (0:00).
4. No fim do guião, esperar que o labeler escreva `MERGE INFERENCIA GRAVADO` em todos os nós.

**No fim da sessão**
```bash
tools/fetch_results.sh sessao_<data>
python3 analysis/make_figures.py experiments/<pasta> --run-filter cenarioA
```

---

## 6. Registo de eventos (ground truth humano)

Durante cada run, alguém (ou o próprio ator com um comando) regista o início e o fim de cada
ação. Como os relógios dos Pis estão acertados pelo do Mac, estes tempos alinham-se diretamente
com os dados.

Isto dá um **segundo ground truth independente das regras** — o que um revisor vai pedir:
- GT das regras (atual): "o que as regras dizem que as câmaras viram";
- GT humano: "o que realmente aconteceu no shop floor".

Comparar os dois mede também a qualidade das próprias regras (que entra na discussão do artigo).

### Como registar (`tools/event_logger.py`, no Mac)

```bash
python3 tools/event_logger.py docs/cenarios/cenario_A.csv --tag cenarioA_rep1
```

- `--tag`: **a mesma etiqueta escrita na UI** (é assim que a análise liga o registo ao run).
- Iniciar a inferência na UI → carregar Enter no instante 0:00 do guião.
- O terminal mostra a ação atual, a instrução e a seguinte. **Enter** no fim de cada ação.
- `n <texto>` + Enter: nota com o instante (ex: `n rasp1 desligado` no cenário B).
- `q` + Enter: abortar (o que já foi registado fica gravado).
- `--auto`: avança sozinho pelos tempos do guião (o ator segue o relógio; menos preciso).

Grava em `experiments/event_logs/<data_hora>_<tag>.csv`. O `analysis/make_figures.py` usa-o
automaticamente (§ "Ground truth humano" do [analysis.md](analysis.md)).

> `experiments/` não vai para o git: fazer cópia de segurança dos registos (são dados da experiência).

---

## 7. Quanto chega?

- **Eventos, não frames:** frames seguidos do mesmo evento estão correlacionados. A unidade de
  análise para estatística é o **evento** (ou o run).
- Cenário A × 5 repetições → 10 eventos por classe de perigo em cada modo (5 com 2 câmaras + 5 com 1).
- Com 5 runs por modo, as barras de erro (IC 95%) da fig. 1 passam a ser informativas.
