"""Multi-scenario robustness check for the trophy forecaster.

Runs the synthetic validation (test_synth.py's simulation) under several
different launch shapes, so the model is not tuned to one made-up week:

  A: base case (as in test_synth.py)
  B: faster player flood (tracked count grows ~2x faster) + slower trophy
     earning (all taus 1.5x) -- heavy dilution, rates rise less
  C: slower player flood + faster earning (taus 0.7x) -- rates rise more

Scores the counts model (main.py), persistence and the log-time trend
baseline on each, and reports the mean pinball loss.  The model must beat
both house baselines on average, not just on one scenario.

Run:  python3 test_robust.py [spread_base]
"""

import json
import math
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_synth as ts


def simulate_variant(n_exp, n_tau, tau_scale, rinf_scale, seed):
    """Like ts.simulate but with scaled dynamics."""
    trophies = json.loads((ts.KIT / "input/p00.in").read_text())["trophies"]
    ids = [t["id"] for t in trophies]
    hours = list(range(0, 169))
    jit = random.Random(99)
    params = {}
    for i in ids:
        rinf, tau = ts.PARAMS[i]
        params[i] = (rinf * jit.uniform(0.85, 1.15),
                     tau * jit.uniform(0.8, 1.25))

    def tracked(h):
        return 638.0 * (1.0 + h / n_tau) ** n_exp

    cohort = []
    prev = tracked(0)
    for h in hours:
        n = tracked(h)
        cohort.append(max(n - prev, 0.0))
        prev = n
    cohort[0] = tracked(0)
    N = [tracked(h) for h in hours]
    E = {i: [0.0] * len(hours) for i in ids}
    for idx, h in enumerate(hours):
        for i in ids:
            rinf, tau = params[i]
            rinf = min(0.95, rinf * rinf_scale)
            tau = tau * tau_scale
            total = 0.0
            for s in range(0, idx + 1):
                age = h - s
                total += cohort[s] * rinf * (1.0 - math.exp(-age / tau))
            E[i][idx] = total
    return trophies, hours, N, E


def run_scenario(name, n_exp, n_tau, tau_scale, rinf_scale, seed, spread_base=None):
    rng = random.Random(seed)
    trophies, hours, N, E = simulate_variant(n_exp, n_tau, tau_scale, rinf_scale, seed)
    case, truth = ts.build_case(trophies, hours, N, E, rng)
    inp = Path("test_in")
    out = Path("test_out")
    inp.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)
    (inp / "f00.in").write_text(json.dumps(case))
    env = dict(__import__("os").environ)
    env["MAKEFUN_INPUT_DIR"] = str(inp)
    env["MAKEFUN_OUTPUT_DIR"] = str(out)
    if spread_base is not None:
        # patch the spread base for this run via env override is not wired;
        # instead we rely on main.py's constant, so just note it.
        pass
    subprocess.run([sys.executable, "main.py"], env=env, check=True, capture_output=True)
    counts = json.loads((out / "f00.out").read_text())
    pers = ts.persistence(case)
    trend = ts.logtime_trend(case)
    lc, _ = ts.score(counts, truth)
    lp, _ = ts.score(pers, truth)
    lt, _ = ts.score(trend, truth)
    cov = sum(1 for r in counts["trophies"] if r["q"][0] <= truth[r["id"]] <= r["q"][4]) / 55
    print(f"{name}: counts={lc:.4f}  persistence={lp:.4f}  trend={lt:.4f}  coverage={cov:.2f}  "
          f"N_target={N[168]:,.0f}")
    return lc, lp, lt


def main():
    scenarios = [
        ("A base      ", 1.2, 4.0, 1.0, 1.0, 20261120),
        ("B flood+slow", 1.5, 3.0, 1.5, 0.9, 20261121),
        ("C slow+fast ", 1.0, 6.0, 0.7, 1.1, 20261122),
    ]
    tot = [0.0, 0.0, 0.0]
    for sc in scenarios:
        r = run_scenario(*sc)
        for k in range(3):
            tot[k] += r[k]
    n = len(scenarios)
    print(f"\nMEAN: counts={tot[0]/n:.4f}  persistence={tot[1]/n:.4f}  trend={tot[2]/n:.4f}")
    print("counts beats trend on average:", tot[0] < tot[2])
    print("counts beats persistence on average:", tot[0] < tot[1])


if __name__ == "__main__":
    main()
