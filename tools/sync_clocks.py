#!/usr/bin/env python3
"""
Mede e acerta o relógio de cada Raspberry pelo relógio do Mac (rede offline, sem NTP).
Necessário para juntar dados de nós diferentes pelo timestamp real.
Modo de uso: docs/deploy.md (secção "Sincronizar relógios")
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NODES_FILE = ROOT / "build" / "nodes.txt"

# Iguais ao tools/deploy.sh. ControlMaster mantém a ligação SSH aberta, para cada
# comando demorar poucos ms (e a medição de offset ser precisa).
SSH_OPTS = [
    "-F", "/dev/null",
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "ControlMaster=auto",
    "-o", "ControlPath=/tmp/gossip-ssh-%r@%h:%p",
    "-o", "ControlPersist=60",
]

MAX_OFFSET_OK = 0.05  # 50 ms: bem abaixo da tolerância de 500 ms usada no merge


# Imagem que existe em todos os Pis (tem python3 e date). Usada quando o sudo pede password.
DOCKER_IMAGE = "p2p-rpi:latest"


def ssh(target, command):
    return subprocess.run(["ssh", *SSH_OPTS, target, command],
                          capture_output=True, text=True, stdin=subprocess.DEVNULL)


def open_master(target):
    """
    Abre a ligação SSH partilhada (ControlMaster) em background, SEM herdar o stdout:
    se fosse criada por um ssh(...) normal, o processo em background ficava agarrado ao
    pipe e o subprocess esperava ~60 s (ControlPersist) -> o script parecia pendurado.
    """
    subprocess.run(["ssh", *SSH_OPTS, "-M", "-N", "-f", target], timeout=20,
                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def set_clock(target, offset, rtt):
    """
    Acerta o relógio do Pi. Devolve o método usado ("sudo" ou "docker").
      1. sudo -n date: hora absoluta do Mac, compensando a latência (rtt/2)
      2. se o sudo pedir password: contentor com CAP_SYS_TIME (o utilizador está no grupo
         docker). Correção RELATIVA (hora do Pi - offset), calculada no próprio Pi no
         instante em que corre -> o arranque do contentor (~1-2 s) não afeta a precisão.
    """
    new_time = time.time() + rtt / 2
    result = ssh(target, f"sudo -n date -s @{new_time:.6f} > /dev/null")
    if result.returncode == 0:
        return "sudo"

    py = ("import time,subprocess; "
          f"t = time.time() - ({offset:.6f}); "
          "subprocess.run(['date', '-s', '@' + format(t, '.6f')], check=True, stdout=subprocess.DEVNULL)")
    cmd = f'docker run --rm --cap-add SYS_TIME --entrypoint python3 {DOCKER_IMAGE} -c "{py}"'
    result = ssh(target, cmd)
    if result.returncode == 0:
        return "docker"
    raise RuntimeError(
        f"sem sudo sem password e o Docker falhou ({result.stderr.strip()[-200:]}). "
        "Ver docs/deploy.md, secção 7."
    )


def fmt(seconds):
    """Offset legível: dias, segundos ou milissegundos."""
    if abs(seconds) >= 86400:
        return f"{seconds / 86400:+.1f} dias"
    if abs(seconds) >= 1:
        return f"{seconds:+.2f} s"
    return f"{seconds * 1000:+.1f} ms"


def measure(target):
    """
    Devolve (offset, rtt) em segundos. offset > 0 = o Pi está adiantado em relação ao Mac.
    O instante do Pi é comparado com o ponto médio do pedido (t0 + t1) / 2.
    """
    t0 = time.time()
    result = ssh(target, "date +%s.%N")
    t1 = time.time()
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return float(result.stdout) - (t0 + t1) / 2, t1 - t0


def main():
    check_only = "--check" in sys.argv
    selected = [a for a in sys.argv[1:] if not a.startswith("--")]

    subprocess.run([sys.executable, str(ROOT / "tools" / "gen_configs.py")],
                   stdout=subprocess.DEVNULL, check=True)

    print(f"{'nó':8s} {'offset antes':>14s} {'offset depois':>14s} {'rtt':>8s}  método")
    failed = []
    for line in NODES_FILE.read_text().splitlines():
        name, ip, user, _ = line.split()
        if selected and name not in selected:
            continue
        target = f"{user}@{ip}"
        try:
            open_master(target)
            measure(target)  # aquecimento (não conta)
            before, rtt = measure(target)
            after, method = before, "-"

            if not check_only:
                method = set_clock(target, before, rtt)
                after, rtt = measure(target)

            status = "✅" if abs(after) <= MAX_OFFSET_OK else "⚠️ "
            print(f"{name:8s} {fmt(before):>14s} {fmt(after):>14s} "
                  f"{rtt * 1000:6.1f}ms  {method:6s} {status}")
        except Exception as e:
            print(f"{name:8s} ❌ {e}")
            failed.append(name)
        finally:
            ssh_exit = ["ssh", *SSH_OPTS, "-O", "exit", target]
            subprocess.run(ssh_exit, capture_output=True, stdin=subprocess.DEVNULL)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
