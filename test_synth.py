"""Synthetic validation for the trophy forecaster.

Builds a realistic fake launch week (a "final-like" case f00) from a
population simulation, then scores three forecasters against the known
truth with the contest's own pinball loss on the logit scale:

  * persistence  -- the house baseline (starter/main.py): last rate, random-walk spread
  * logtime trend -- the house curve baseline: least-squares line of logit(rate)
                     against ln(1 + hours since launch) over the last 24h
  * counts model -- this repo's main.py: separate earner / tracked forecasts

The simulation (grounded, not arbitrary):
  * Tracked players N(t) grow as a power law, 638 at launch to ~40k at the
    deadline (2.5 days) and ~150k at the target (7 days).  Newzoo's
    37-51M week-one copies say the flood of new players is real.
  * Players join in hourly cohorts.  A cohort of age `a` has earned trophy i
    with probability R_inf_i * (1 - exp(-a / tau_i)): early story trophies
    are fast (tau ~18h, R_inf ~0.65), the final mission and grinds are
    slow (tau 120-500h, R_inf 0.02-0.25), the platinum tracks the rarest.
    This is the same shape GTA V shows over its lifetime (opening story
    trophy ~51-93%, mid-story ~20-60%, final mission ~19-35%, platinum
    ~4-6%), compressed into week one.
  * Exophase only refreshes every few hours: recorded reads freeze between
    refreshes, so the history shows the real step pattern (tracked count
    jumps, rates dip by dilution) that the RULES describe on launch night.

Run:  python3 test_synth.py
"""

import json
import math
import random
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT = Path.home() / "workspace/makefun/gta6-trophy-rarity-forecast-01/gta6-trophy-rarity-forecast-01-kit"

LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
DEADLINE = datetime(2026, 11, 21, 11, 30, 0, tzinfo=timezone.utc)
TARGET = datetime(2026, 11, 26, 0, 0, 0, tzinfo=timezone.utc)

# (trophy ids) -> (R_inf, tau_hours).  Categories from the trophy list.
PARAMS = {}
for i in (3,):
    PARAMS[i] = (0.65, 18)
for i in (4,):
    PARAMS[i] = (0.55, 30)
for i in (5,):
    PARAMS[i] = (0.45, 48)
for i in (6,):
    PARAMS[i] = (0.38, 72)
PARAMS[7] = (0.25, 120)      # final mission (gold)
PARAMS[14] = (0.02, 500)     # 100% completion (gold)
PARAMS[1] = (0.015, 500)     # platinum
PARAMS[2] = (0.75, 8)        # Gone Fishing (early activity)
for i in (8, 9, 10, 11, 12, 13):
    PARAMS[i] = (0.45, 36)
for i in (22, 23, 24, 26, 27):
    PARAMS[i] = (0.35, 60)
for i in (16, 18, 19, 20, 21, 28, 29, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55):
    PARAMS[i] = (0.20, 120)
for i in (15, 17, 25, 30, 31, 40):
    PARAMS[i] = (0.10, 300)


def tracked(t_hours):
    """True tracked-player count: power law from 638 at launch."""
    return 638.0 * (1.0 + t_hours / 4.0) ** 1.2


def simulate():
    """Hourly cohort simulation. Returns (times, N, E) with E[t][i] earners."""
    import random as _random
    # Small per-trophy jitter so grouped trophies do not share one identical
    # curve (a real launch has one curve per trophy, not per category).
    jit = _random.Random(99)
    trophies = json.loads((KIT / "input/p00.in").read_text())["trophies"]
    ids = [t["id"] for t in trophies]
    params = {}
    for i in ids:
        rinf, tau = PARAMS[i]
        params[i] = (rinf * jit.uniform(0.85, 1.15),
                     tau * jit.uniform(0.8, 1.25))
    hours = list(range(0, 169))  # 0..168h (launch to target)
    # cohort sizes per hour
    cohort = []
    prev = tracked(0)
    for h in hours:
        n = tracked(h)
        cohort.append(max(n - prev, 0.0))
        prev = n
    cohort[0] = tracked(0)  # the launch cohort
    N = [tracked(h) for h in hours]
    E = {i: [0.0] * len(hours) for i in ids}
    for idx, h in enumerate(hours):
        for i in ids:
            rinf, tau = params[i]
            total = 0.0
            for s in range(0, idx + 1):
                age = h - s
                total += cohort[s] * rinf * (1.0 - math.exp(-age / tau))
            E[i][idx] = total
    return trophies, hours, N, E


def build_case(trophies, hours, N, E, rng):
    """Build the f00 case: reads every 30 min early then 10 min, frozen
    between Exophase refreshes (every ~4h), rates rounded to 2 decimals."""
    # refresh times (hours since launch): 0, then every ~4h with jitter
    refresh = [0.0]
    t = 0.0
    while t < 60:
        t += rng.uniform(3.0, 6.0)
        refresh.append(t)
    refresh.append(59.5)
    # true rate at any hour (linear interp on the hourly grid)
    def true_rate(i, h):
        lo = max(0, min(167, int(math.floor(h))))
        hi = min(168, lo + 1)
        f = h - lo
        r_lo = E[i][lo] / N[lo] * 100.0
        r_hi = E[i][hi] / N[hi] * 100.0
        return r_lo + (r_hi - r_lo) * f

    def true_n(h):
        lo = max(0, min(167, int(math.floor(h))))
        hi = min(168, lo + 1)
        f = h - lo
        return N[lo] + (N[hi] - N[lo]) * f

    # read slots: every 30 min from -0.17h (23:50 18 Nov) to deadline
    slots = []
    h = -1.0 / 6.0
    while h <= 59.5:
        slots.append(h)
        h += 0.5 if h < 24 else (1.0 / 6.0)
    history = []
    cur_n = None
    cur_rates = None
    for h in slots:
        # last refresh at or before h
        r = max([x for x in refresh if x <= h], default=0.0)
        n = true_n(r)
        rates = [round(true_rate(i, r), 2) for i in [t["id"] for t in trophies]]
        if cur_n is None or n != cur_n or rates != cur_rates:
            cur_n, cur_rates = n, rates
        slot_dt = LAUNCH + timedelta(hours=h)
        history.append({
            "slot": slot_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "at": slot_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "players_tracked": int(round(cur_n)),
            "trophies_earned": int(round(sum(cur_rates) / 100.0 * cur_n)),
            "rates": cur_rates,
        })
    horizon = int((TARGET - DEADLINE).total_seconds() // 60)
    case = {
        "schema": "makefun/gta6-trophies-case/v1",
        "id": "f00",
        "at": "2026-11-21T11:30:00Z",
        "horizon_minutes": horizon,
        "target": "2026-11-26T00:00:00Z",
        "levels": [0.05, 0.25, 0.5, 0.75, 0.95],
        "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                 "release_date": "2026-11-19",
                 "source": "https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/"},
        "trophies": trophies,
        "history": history,
        "notes": "Synthetic final-like case for local validation only.",
    }
    # truth at target
    truth = {}
    for t in trophies:
        i = t["id"]
        truth[i] = E[i][168] / N[168] * 100.0
    return case, truth


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    return math.log(clamp(pct) / (100.0 - clamp(pct)))


def pinball(y, q, tau):
    d = logit(y) - logit(q)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def score(forecast, truth):
    levels = [0.05, 0.25, 0.5, 0.75, 0.95]
    tot, n = 0.0, 0
    per = {}
    for row in forecast["trophies"]:
        y = truth[row["id"]]
        for tau, q in zip(levels, row["q"]):
            tot += pinball(y, q, tau)
            n += 1
        per[row["id"]] = abs(y - row["q"][2])
    return tot / n, sum(per.values()) / len(per)


def persistence(case):
    """House baseline (starter): last rate, random-walk spread 0.5*sqrt(d)."""
    import math as m
    last = case["history"][-1]
    d = case["horizon_minutes"] / 1440.0
    spread = 0.5 * m.sqrt(d)
    Z = [-1.645, -0.674, 0, 0.674, 1.645]
    rows = []
    for i, tr in enumerate(case["trophies"]):
        r = last["rates"][i]
        lg = logit(r if r is not None else 0)
        rows.append({"id": tr["id"], "q": [
            round(min(99.99, max(0.01, 100 / (1 + m.exp(-(lg + z * spread))))), 4)
            for z in Z]})
    return {"trophies": rows}


def logtime_trend(case):
    """House curve baseline: least-squares line of logit(rate) vs
    ln(1 + hours since launch) over the last 24h, read at the target;
    spread = random walk + residual."""
    import math as m
    launch = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
    hist = case["history"]
    last = hist[-1]
    t_last = (datetime.fromisoformat(last["at"].replace("Z", "+00:00")) - launch).total_seconds() / 3600
    t_tgt = (datetime.fromisoformat(case["target"].replace("Z", "+00:00")) - launch).total_seconds() / 3600
    x_tgt = m.log(1 + t_tgt)
    d = case["horizon_minutes"] / 1440.0
    Z = [-1.645, -0.674, 0, 0.674, 1.645]
    rows = []
    for i, tr in enumerate(case["trophies"]):
        pts = []
        last_r = 0.0
        for rd in hist:
            t = (datetime.fromisoformat(rd["at"].replace("Z", "+00:00")) - launch).total_seconds() / 3600
            if t < t_last - 24:
                continue
            r = rd["rates"][i]
            if r is None:
                r = last_r
            else:
                last_r = r
            pts.append((m.log(1 + max(t, -0.9)), logit(r)))
        # dedupe identical x
        ded = []
        for p in pts:
            if not ded or ded[-1] != p:
                ded.append(p)
        pts = ded
        if len(pts) < 3:
            med = last["rates"][i] if last["rates"][i] is not None else 0
            spread = 0.5 * m.sqrt(d)
        else:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            sxx = sum((x - mx) ** 2 for x in xs)
            slope = sum((x - mx) * (y - my) for x, y in pts) / sxx if sxx > 0 else 0.0
            inter = my - slope * mx
            med_logit = inter + slope * x_tgt
            resid = m.sqrt(sum((y - (inter + slope * x)) ** 2 for x, y in pts) / len(pts))
            spread = m.sqrt((0.5 * m.sqrt(d)) ** 2 + resid ** 2)
            med = 100 / (1 + m.exp(-med_logit))
        med = clamp(med)
        lg = logit(med)
        rows.append({"id": tr["id"], "q": [
            round(min(99.99, max(0.01, 100 / (1 + m.exp(-(lg + z * spread))))), 4)
            for z in Z]})
    return {"trophies": rows}


def main():
    rng = random.Random(20261120)
    trophies, hours, N, E = simulate()
    case, truth = build_case(trophies, hours, N, E, rng)
    print(f"synthetic f00: {len(case['history'])} reads, "
          f"N at deadline={case['history'][-1]['players_tracked']:,}, "
          f"N at target={N[168]:,.0f}")
    print(f"truth rate range: {min(truth.values()):.2f}% .. {max(truth.values()):.2f}%")
    inp = Path("test_in")
    out = Path("test_out")
    inp.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)
    (inp / "f00.in").write_text(json.dumps(case))
    # counts model (this repo)
    env = dict(__import__("os").environ)
    env["MAKEFUN_INPUT_DIR"] = str(inp)
    env["MAKEFUN_OUTPUT_DIR"] = str(out)
    subprocess.run([sys.executable, "main.py"], env=env, check=True)
    counts = json.loads((out / "f00.out").read_text())
    pers = persistence(case)
    trend = logtime_trend(case)
    for name, fc in [("counts model (main.py)", counts),
                     ("persistence (house baseline)", pers),
                     ("log-time trend (house curve)", trend)]:
        loss, miss = score(fc, truth)
        print(f"{name:32s} pinball loss={loss:.6f}  typical miss={miss:.2f} pts")
    # coverage of the counts model
    cov = sum(1 for r in counts["trophies"] if r["q"][0] <= truth[r["id"]] <= r["q"][4]) / 55
    print(f"counts model 90% coverage: {cov:.2f} (honest ~0.90)")
    # a few rows for the eye
    print("\nsample rows (id, truth, q05, q50, q95):")
    names = {t["id"]: t["name"] for t in trophies}
    for i in (3, 4, 6, 7, 14, 15, 1):
        r = [x for x in counts["trophies"] if x["id"] == i][0]
        print(f"  {i:2d} {names[i][:32]:32s} truth={truth[i]:6.2f}%  "
              f"q05={r['q'][0]:6.2f} q50={r['q'][2]:6.2f} q95={r['q'][4]:6.2f}")


if __name__ == "__main__":
    main()
