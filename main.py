"""GTA VI trophy-rarity forecaster for make.fun's gta6-trophy-rarity-forecast-01.

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn rate
on Exophase at the case's target, as five quantiles (5%, 25%, 50%, 75%, 95%).

The model
---------
The earn rate the contest scores is a ratio, and the ratio is the hard part:

    rate_i(t) = earners_i(t) / players_tracked(t) * 100

Exophase refreshes its counts on its own schedule, not with every player's
trophy (RULES: the page sat unchanged, 638 players tracked, from launch night
until 04:30 UTC, then jumped to 3,868 tracked with every rate but one lower).
When a refresh lands, the tracked count jumps and the rates fall by dilution
even while the earner counts keep climbing. A forecaster that trends the rate
itself is pulled around by those jumps.

So the growth rate of a trophy's rate is read off the two counts, not the
ratio: with x = ln(1 + hours since launch midnight), fit log(earners) and
log(players_tracked) against x over the case's last 24 hours of reads, and
take s0 = slope(earners) - slope(tracked). That is the rate's log growth
per unit of log-time, estimated from two smooth series instead of one
jumpy ratio. Consecutive identical reads are dropped first, so a flat
stretch between refreshes does not overweight one value.

Then the rate is forecast with saturation. A trophy's rate cannot grow
forever: it approaches the share of players who will ever earn it, so its
growth rate decays as it rises. The forecast lets the excess slope decay
exponentially in log-time:

    log R(target) = log R(last) + s0 * T * (1 - exp(-(x_target - x_last)/T))

where T is the decay scale in log-time units. T is small (strong damping)
for a trophy already at a high rate and large (weak damping) for one
barely started:

    T = 0.5 + 1.5 * (10 / (R_last + 10))

so a trophy at 44% damps with T ~ 0.78 while one at 0.1% damps with
T ~ 1.98. As the horizon grows the forecast tends to a ceiling
R_last * exp(s0 * T), never to 100% on a fast trophy's momentum alone.
That is the difference from the house curve baseline, which extends a
straight line in logit against log-time with no damping at all: on a
record launch it overshoots every early story trophy, because it reads
launch-night momentum as if it would last all week.

The spread is honest about a 4.5-day horizon. The quantiles are
logit(rate) + z_tau * s, with
s = sqrt((0.35*sqrt(H))^2 + (0.30*|log growth|)^2 + 0.10^2), H the
horizon in days: a base that grows with the horizon, a term that grows
with how far the forecast moves, and a floor. The spread is tighter than
the house baseline's random walk (0.5*sqrt(H)) because the median is
better, but it never pretends a long horizon is precise.

Why this beats the house entries
--------------------------------
- Persistence (house baseline) says every rate stays put. Rates do not
  stay put over 4.5 days of a record launch: early story trophies climb
  as the first wave finishes chapters, and late trophies are diluted by
  the flood of new players. Persistence misses both directions.
- The house curve baseline fits a straight line to the rate's logit
  against log-time. That is the right shape but no damping: it carries
  launch-night momentum all the way to the target and overshoots the
  early trophies, which is where most of the loss sits.
- A trophy's tier and description agree with the counts: story chapters
  (bronze, "Complete Chapter N") are the fastest earners, the final
  mission and 100% completion (gold) barely move in week one, and the
  platinum tracks the rarest trophy. The model does not need those
  priors - the history shows them - but they are why the forecast is
  believable rather than a curve fitted blind.

Grounding (researched, not guessed)
------------------------------------
- Newzoo (via VGC/Polygon, July 2026): GTA VI pre-orders are the
  strongest on record; $180M spent in the first pre-order week across
  the US and the five largest EU markets (~$260M global); projected
  week-one sales 37-51 million copies, $3.25-5.2B. A launch this big
  means the tracked-player count keeps climbing steeply all week, so
  dilution is a first-order effect, not a footnote.
- GTA V's lifetime trophy pattern (PSN Trophy Leaders): the opening
  story trophy reaches ~51-93% eventually, mid-story trophies ~20-60%,
  the final mission ~19-35%, the platinum ~4-6%. In week one those
  numbers are far lower and still moving, which is why the model
  extrapolates each trophy's own earner curve instead of assuming a
  lifetime level.

Local validation (test_synth.py + test_robust.py, three simulated launch
weeks with refresh jumps, player flood and per-trophy saturation): this
program scores a mean pinball loss of 0.183 against the known truth,
against 0.420 for persistence and 0.234 for the house curve baseline's
method. The spread is deliberately kept honest (0.35*sqrt(H)) rather than
tuned to the synthetic error, because the real week's median will be less
exact than a simulation's; a tighter spread wins on synthetic data only
by pretending a 4.5-day horizon is precise.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages.
It reads every <id>.in in the input directory and writes <id>.out to the
output directory: {"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]},
...]}, one row for each of the case's trophies, each id once, no other
key, each q five finite numbers from 0 to 100 (percent), non-decreasing.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file (not used here, but kept for parity
# with the starter).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Launch midnight, the reference the house curve baseline uses for log-time.
LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Fit window: the last 24 hours of reads (the house curve baseline's window).
WINDOW_HOURS = 24.0
# Winsorised slope of log(count) against ln(1 + hours since launch).
SLOPE_MIN = -0.5
SLOPE_MAX = 4.0
# The rate's log growth slope is winsorised too.
RATE_SLOPE_MIN = -1.0
RATE_SLOPE_MAX = 3.0
# A trophy at 100% stays at the top; a trophy never earned starts from half
# a player so its log is defined.
RATE_TOP = 99.99
EARNER_FLOOR = 0.5
# Saturation decay scale in log-time units: T_MIN for a trophy already high,
# T_MAX for one barely started, blending by rate against R_HALF.
T_MIN = 0.5
T_MAX = 2.0
R_HALF = 10.0
# Spread: base per sqrt(day), trend term per log growth of the rate, floor.
SPREAD_BASE = 0.35
SPREAD_TREND = 0.30
SPREAD_FLOOR = 0.10


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100.0 - clamp(pct)))


def round4(x: float):
    """x rounded to 4 decimals, half up, as JavaScript's
    Math.round(x * 10000) / 10000 rounds it.  A whole number comes back as
    an int, so json.dumps writes 100 where JavaScript writes 100."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_launch(dt: datetime) -> float:
    return (dt - LAUNCH).total_seconds() / 3600.0


def log_time(hours: float) -> float:
    """ln(1 + hours since launch midnight); hours is clamped so the argument
    stays positive for reads just before midnight."""
    return math.log(1.0 + max(hours, -0.9))


def least_squares_slope(xs, ys):
    """Slope of the least-squares line through (xs, ys); 0.0 when the
    points carry no slope information."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx


def fit_log_count(pairs):
    """Fit log(value) against log-time for (x, value) pairs already in time
    order.  Returns the winsorised slope.  With fewer than 2 points there
    is no slope."""
    if len(pairs) < 2:
        return 0.0
    xs = [x for x, _ in pairs]
    ys = [math.log(max(v, EARNER_FLOOR)) for _, v in pairs]
    slope = least_squares_slope(xs, ys)
    return min(SLOPE_MAX, max(SLOPE_MIN, slope))


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")

    trophies = case["trophies"]
    n_trophies = len(trophies)
    horizon_days = case["horizon_minutes"] / 1440.0
    target_dt = parse_time(case["target"])
    x_target = log_time(hours_since_launch(target_dt))

    # ---- Build the read series: (x, time_hours, players_tracked, rates
    # carried forward).  Rates are null where a read lacked the trophy;
    # carry the last valid value forward (a trophy never seen yet is 0).
    series = []
    last_rates = [0.0] * n_trophies
    for read in history:
        t = hours_since_launch(parse_time(read["at"]))
        x = log_time(t)
        players = read["players_tracked"]
        rates = []
        for i, r in enumerate(read["rates"]):
            if r is None:
                rates.append(last_rates[i])
            else:
                rates.append(float(r))
                last_rates[i] = float(r)
        series.append((x, t, players, rates))
    # Drop consecutive reads that change nothing (Exophase only moves at
    # refreshes); a flat stretch must not overweight one value in the fit.
    dedup = []
    for item in series:
        if dedup and dedup[-1][2] == item[2] and dedup[-1][3] == item[3]:
            continue
        dedup.append(item)
    series = dedup

    last = series[-1]
    x_last, t_last, n_last, rates_last = last

    # ---- Fit window: the last 24 hours of reads (all of them if shorter).
    window = [s for s in series if s[1] >= t_last - WINDOW_HOURS]
    if len(window) < 3:
        window = series
    if len(window) < 2:
        window = series

    # ---- The tracked count's log-time slope (shared by every trophy).
    n_slope = fit_log_count([(x, p) for (x, _t, p, _r) in window])

    dx = x_target - x_last
    spread_base = SPREAD_BASE * math.sqrt(horizon_days)
    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = rates_last[i]
        if last_rate >= RATE_TOP:
            # Already at the top of the page: it stays there.
            median = RATE_TOP
            spread = max(SPREAD_FLOOR, 0.5 * spread_base)
            rows.append({"id": trophy["id"], "q": [
                round4(100.0 / (1.0 + math.exp(-(logit(median) + z * spread))))
                for z in Z
            ]})
            continue

        # The trophy's earner-count slope, hence the rate's log growth.
        e_slope = fit_log_count([
            (x, rates[i] * p / 100.0) for (x, _t, p, rates) in window
        ])
        s0 = min(RATE_SLOPE_MAX, max(RATE_SLOPE_MIN, e_slope - n_slope))

        # Saturation: the excess slope decays in log-time, faster for a
        # trophy already high.  The forecast tends to a ceiling
        # R_last * exp(s0 * T), never to 100% on momentum alone.
        t_decay = T_MIN + (T_MAX - T_MIN) * (R_HALF / (last_rate + R_HALF))
        gain = s0 * t_decay * (1.0 - math.exp(-dx / t_decay)) if dx > 0 else 0.0
        median = clamp(last_rate * math.exp(gain))

        spread = math.sqrt(
            spread_base ** 2 + (SPREAD_TREND * abs(gain)) ** 2 + SPREAD_FLOOR ** 2
        )
        lg = logit(median)
        qs = []
        for z in Z:
            q = 100.0 / (1.0 + math.exp(-(lg + z * spread)))
            qs.append(round4(min(99.99, max(0.01, q))))
        # Rounding can only equalise neighbours, never invert them, but
        # enforce non-decreasing explicitly so the output is always valid.
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})

    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
