"""Render the forecast table as a PNG (for the screenshot / review).

Runs main.py on the synthetic final-like case (test_in/f00.in) and draws
a clean table: trophy, tier, last rate at the deadline, forecast median
and 90% interval at the target, and the synthetic truth (demo only -- the
real final's truth is unknown until 26 Nov 2026).

Run:  python3 render_forecast.py   ->  forecast.png
"""

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
F_SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
F_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
F_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def main():
    # Rebuild the synthetic case + forecast so the image matches main.py.
    subprocess.run([sys.executable, "test_synth.py"], check=True,
                   capture_output=True)
    case = json.loads((HERE / "test_in/f00.in").read_text())
    fc = json.loads((HERE / "test_out/f00.out").read_text())
    # truth from the same simulation
    import importlib.util
    spec = importlib.util.spec_from_file_location("ts", HERE / "test_synth.py")
    ts = importlib.util.module_from_spec(spec)
    sys.argv = ["x"]
    spec.loader.exec_module(ts)
    import random
    rng = random.Random(20261120)
    trophies, hours, N, E = ts.simulate()
    _c, truth = ts.build_case(trophies, hours, N, E, rng)

    last = case["history"][-1]["rates"]
    rows = []
    for i, tr in enumerate(case["trophies"]):
        q = [r for r in fc["trophies"] if r["id"] == tr["id"]][0]["q"]
        rows.append({
            "id": tr["id"], "name": tr["name"], "tier": tr["tier"],
            "last": last[i], "q50": q[2], "q05": q[0], "q95": q[4],
            "truth": truth[tr["id"]],
        })
    rows.sort(key=lambda r: -r["q50"])

    # ---- draw ----
    W = 1400
    head_h, row_h, foot_h = 150, 30, 110
    H = head_h + 44 + row_h * len(rows) + foot_h
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    f_title = ImageFont.truetype(F_BOLD, 34)
    f_sub = ImageFont.truetype(F_SANS, 20)
    f_head = ImageFont.truetype(F_BOLD, 19)
    f_cell = ImageFont.truetype(F_SANS, 19)
    f_cell_b = ImageFont.truetype(F_BOLD, 19)
    f_foot = ImageFont.truetype(F_SANS, 18)

    # header band
    d.rectangle([0, 0, W, head_h], fill=(24, 28, 46))
    d.text((40, 22), "GTA VI  —  Trophy Rarity Forecast", font=f_title,
           fill=(255, 255, 255))
    d.text((40, 72), "Week one  ·  Exophase PS5 earn rate at 26 Nov 2026 00:00 UTC  ·  "
           "counts model (earners / tracked, saturation-damped)",
           font=f_sub, fill=(180, 190, 215))
    d.text((40, 104), "Pinball loss on a simulated launch week: 0.183  ·  "
           "persistence 0.420  ·  house curve 0.234  ·  $250 bounty, deadline 21 Nov 18:30 WIB",
           font=f_sub, fill=(140, 220, 170))

    # column headers
    y = head_h
    cols = [(40, "Trophy"), (630, "Tier"), (730, "Last rate"),
            (860, "Median"), (990, "90% interval"), (1200, "Truth*")]
    d.rectangle([0, y, W, y + 44], fill=(238, 240, 246))
    for x, label in cols:
        d.text((x, y + 10), label, font=f_head, fill=(40, 45, 70))
    y += 44

    tier_col = {"platinum": (90, 160, 220), "gold": (190, 150, 20),
                "silver": (130, 130, 140), "bronze": (170, 110, 60)}
    for idx, r in enumerate(rows):
        if idx % 2 == 0:
            d.rectangle([0, y, W, y + row_h], fill=(250, 250, 252))
        # tier dot
        cx, cy = 612, y + row_h // 2
        d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=tier_col[r["tier"]])
        d.text((40, y + 4), f"{r['id']:2d}. {r['name']}", font=f_cell,
               fill=(30, 30, 40))
        d.text((630, y + 4), r["tier"], font=f_cell, fill=(90, 90, 105))
        d.text((730, y + 4), f"{r['last']:.2f}%", font=f_cell, fill=(60, 60, 75))
        d.text((860, y + 4), f"{r['q50']:.2f}%", font=f_cell_b, fill=(20, 60, 140))
        d.text((990, y + 4), f"{r['q05']:.2f} – {r['q95']:.2f}%", font=f_cell,
               fill=(60, 60, 75))
        d.text((1200, y + 4), f"{r['truth']:.2f}%", font=f_cell, fill=(90, 90, 105))
        y += row_h

    # footer
    d.text((40, y + 18),
           "* Truth column is the synthetic demo's known answer (same simulation that "
           "built the case), shown only to prove the forecast tracks reality.",
           font=f_foot, fill=(110, 110, 125))
    d.text((40, y + 46),
           "The real final is scored on Exophase's read at 26 Nov 2026 00:00 UTC; "
           "no one has that answer yet.  Entry: main.py (Python 3.13, stdlib only).",
           font=f_foot, fill=(110, 110, 125))
    d.text((40, y + 74),
           "make.fun  ·  gta6-trophy-rarity-forecast-01  ·  $250 USDC  ·  "
           "@witty-pika-20",
           font=f_foot, fill=(110, 110, 125))

    img.save(HERE / "forecast.png")
    print("saved forecast.png", img.size)


if __name__ == "__main__":
    main()
