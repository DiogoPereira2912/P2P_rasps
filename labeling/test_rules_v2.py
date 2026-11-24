#!/usr/bin/env python3
"""
Teste das regras v2 (labeling/rules_v2.yaml) com poses sintéticas: cada gesto deve disparar
SÓ a regra certa, e as "armadilhas" não devem disparar nada — perto e longe da câmara.
Correr depois de mudar limiares: python3 labeling/test_rules_v2.py
"""
import math
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labeling_utils import body_conditions_ok, frame_level

RULES = yaml.safe_load(open(Path(__file__).resolve().parent / "rules_v2.yaml"))

# Pose de referência de pé, de frente para a câmara (tronco = 250 px; y cresce para baixo)
STANDING = {
    0: (0, -200), 5: (-60, -150), 6: (60, -150), 7: (-70, -60), 8: (70, -60),
    9: (-75, 30), 10: (75, 30), 11: (-40, 100), 12: (40, 100),
    13: (-45, 230), 14: (45, 230), 15: (-50, 350), 16: (50, 350),
}


def rotate(points, ids, angle_deg, pivot):
    a = math.radians(angle_deg)
    out = dict(points)
    for i in ids:
        x, y = points[i][0] - pivot[0], points[i][1] - pivot[1]
        out[i] = (pivot[0] + x * math.cos(a) - y * math.sin(a), pivot[1] + x * math.sin(a) + y * math.cos(a))
    return out


def poses():
    p = {}
    p["de pé (normal)"] = STANDING
    p["QUEDA: deitado no chão"] = rotate(STANDING, STANDING.keys(), 90, (0, 0))
    upper = [0, 5, 6, 7, 8, 9, 10]
    p["armadilha: baixar-se p/ apanhar caixa"] = rotate(STANDING, upper, -75, (0, 100))
    p["SOS: dois braços acima da cabeça"] = {**STANDING, 7: (-80, -240), 8: (80, -240), 9: (-60, -330), 10: (60, -330)}
    p["armadilha: levantar só um braço"] = {**STANDING, 7: (-80, -240), 9: (-60, -330)}
    p["BRAÇO ESTENDIDO: braço para o lado"] = {**STANDING, 8: (170, -150), 10: (280, -150)}
    p["armadilha: apontar com braço dobrado"] = {**STANDING, 8: (130, -110), 10: (150, -200)}
    p["armadilha: braços pendurados"] = {**STANDING, 7: (-62, -50), 9: (-64, 50)}
    return p


def to_detection(points, scale, offset):
    """Keypoints [x, y, conf] e bbox, numa dada escala (distância à câmara) e posição."""
    kpts = [[-1, -1, 0.0] for _ in range(17)]
    for i, (x, y) in points.items():
        kpts[i] = [offset[0] + x * scale, offset[1] + y * scale, 0.9]
    xs = [k[0] for k in kpts if k[2] > 0]
    ys = [k[1] for k in kpts if k[2] > 0]
    pad = 15 * scale
    return kpts, (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def fired(kpts, bbox):
    out = []
    for rule in RULES["regras"]:
        cond = rule["condicoes"]
        if body_conditions_ok(kpts, bbox, cond, RULES.get("kpt_conf_min", 0.3)):
            out.append(rule["label"])
    return out


EXPECTED = {
    "de pé (normal)": [], "QUEDA: deitado no chão": ["QUEDA"],
    "armadilha: baixar-se p/ apanhar caixa": [], "SOS: dois braços acima da cabeça": ["SINAL_SOS"],
    "armadilha: levantar só um braço": [], "BRAÇO ESTENDIDO: braço para o lado": ["BRACO_ESTENDIDO"],
    "armadilha: apontar com braço dobrado": [], "armadilha: braços pendurados": [],
}


def main():
    ok = True
    print(f"{'pose':40s} {'perto (1.0)':22s} {'longe (0.25)':22s}")
    for name, pts in poses().items():
        results = []
        for scale, offset in ((1.0, (900, 500)), (0.25, (300, 200))):
            kpts, bbox = to_detection(pts, scale, offset)
            results.append(fired(kpts, bbox))
        good = all(r == EXPECTED[name] for r in results)
        ok &= good
        print(f"{'✅' if good else '❌'} {name:38s} {str(results[0] or '-'):22s} {str(results[1] or '-'):22s}")

    # Várias pessoas no mesmo frame: a câmara dispara se alguém cumprir a regra
    df = pd.DataFrame({"ts": [1, 1, 1, 2], "QUEDA_rasp1": [0, 1, 0, 0], "SINAL_SOS_rasp1": [0, 0, 0, 1]})
    fl = frame_level(df)
    multi_ok = fl.to_dict("records") == [{"ts": 1, "QUEDA_rasp1": 1, "SINAL_SOS_rasp1": 0},
                                         {"ts": 2, "QUEDA_rasp1": 0, "SINAL_SOS_rasp1": 1}]
    ok &= multi_ok
    print(f"{'✅' if multi_ok else '❌'} várias pessoas no frame -> 1 linha por frame com 'pelo menos uma'")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
