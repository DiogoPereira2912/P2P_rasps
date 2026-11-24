#!/usr/bin/env python3
"""
Registo de eventos durante um run (ground truth humano), a partir do guião do cenário.
Corre no Mac: os relógios dos Pis estão acertados pelo do Mac (tools/sync_clocks.py),
por isso os instantes registados alinham-se diretamente com os dados dos Pis.
Modo de uso: docs/cenario.md (secção "Registo de eventos")
"""
import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "experiments" / "event_logs"

FIELDS = ["tag", "segmento", "acao", "zona", "n_camaras", "esperado",
          "inicio_ts", "fim_ts", "inicio", "fim", "nota"]


def fmt_clock(seconds):
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def load_script(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["duracao"] = float(r["fim_s"]) - float(r["inicio_s"])
    return rows


def show(seg, idx, total, nxt):
    print("\n" + "=" * 64)
    print(f" [{idx + 1}/{total}]  AGORA: {seg['acao']}  (zona {seg['zona']}, "
          f"{seg['n_camaras']} câmara(s), ~{seg['duracao']:.0f} s)")
    print(f"           {seg['instrucao']}")
    if nxt:
        print(f"  a seguir: {nxt['acao']} (zona {nxt['zona']})")
    print("=" * 64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", help="guião, ex: docs/cenarios/cenario_A.csv")
    parser.add_argument("--tag", required=True,
                        help="a MESMA etiqueta escrita na UI (ex: cenarioA_rep1)")
    parser.add_argument("--auto", action="store_true",
                        help="avança sozinho pelos tempos do guião (sem carregar Enter)")
    args = parser.parse_args()

    script = load_script(args.script)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{datetime.now():%Y-%m-%d_%H%M%S}_{args.tag}.csv"

    print(f"Guião: {args.script} ({len(script)} segmentos, "
          f"{fmt_clock(sum(s['duracao'] for s in script))})")
    print(f"Registo: {out_path.relative_to(ROOT)}")
    if args.auto:
        print("Modo automático: segue os tempos do guião.")
    else:
        print("Enter = segmento seguinte | n + Enter = nota (ex: nó desligado) | q + Enter = abortar")
    input("\n▶ Iniciar a inferência na UI e carregar Enter no instante 0:00 do guião...")

    rows, t0 = [], time.time()
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for i, seg in enumerate(script):
            start = time.time()
            nxt = script[i + 1] if i + 1 < len(script) else None
            show(seg, i, len(script), nxt)

            if args.auto:
                time.sleep(seg["duracao"])
            else:
                while True:
                    cmd = input(f"  [{fmt_clock(time.time() - t0)}] Enter quando terminar > ").strip().lower()
                    if cmd == "q":
                        print("Abortado. Segmentos registados até agora ficam no ficheiro.")
                        return
                    if cmd.startswith("n"):
                        note = cmd[1:].strip() or input("  nota: ")
                        now = time.time()
                        writer.writerow({"tag": args.tag, "segmento": -1, "acao": "NOTA",
                                         "inicio_ts": f"{now:.6f}", "fim_ts": f"{now:.6f}",
                                         "inicio": datetime.fromtimestamp(now).isoformat(),
                                         "fim": datetime.fromtimestamp(now).isoformat(), "nota": note})
                        f.flush()
                        print(f"  📝 nota registada: {note}")
                        continue
                    break

            end = time.time()
            writer.writerow({
                "tag": args.tag, "segmento": i, "acao": seg["acao"], "zona": seg["zona"],
                "n_camaras": seg["n_camaras"], "esperado": seg["esperado_min_nos_2"],
                "inicio_ts": f"{start:.6f}", "fim_ts": f"{end:.6f}",
                "inicio": datetime.fromtimestamp(start).isoformat(),
                "fim": datetime.fromtimestamp(end).isoformat(), "nota": "",
            })
            f.flush()  # não perder nada se o run for interrompido
            rows.append(end - start)

    print(f"\n✅ Run registado ({fmt_clock(time.time() - t0)}). Ficheiro: {out_path.relative_to(ROOT)}")
    print("   Parar a inferência na UI quando o labeler gravar o merge final.")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        sys.exit("\nInterrompido. Os segmentos já registados ficam no ficheiro.")
