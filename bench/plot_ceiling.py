"""fig:ceiling from results/bench/sweep/condition_scale.json: x = OminiControl condition_scale (log axis, 0 shown at the left as the
condition-suppressed point), y = canny F1 (tolerant) on the dev-200; VAE decode (512), vanilla PiD at K = 28 and 24 at the 512 view (solid)
and native 2048 (dashed); horizontal lines = the decoder-only ceilings on the same 200 real images (VAE round trip, PiD round trip 512 / 2048).
CGD curves are added later from the same JSON schema (keys cgd_k<K>@512 / @2048) if present.
Usage: python plot_ceiling.py [--out /home/wookiekim/2026-conditional-decoding/figures/ceiling.pdf]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import REPO_BENCH

ap = argparse.ArgumentParser()
ap.add_argument("--json", default=str(REPO_BENCH.parent / "results" / "bench" / "sweep" / "condition_scale.json"))
ap.add_argument("--out", default="/home/wookiekim/2026-conditional-decoding/figures/ceiling.pdf")
ap.add_argument("--k", default="24", help="which PiD truncation to draw as the main generative curve (28 also drawn, lighter)")
a = ap.parse_args()
J = json.load(open(a.json))
scales = sorted(J["scales"]); per = {m["scale"]: m for m in J["per_scale"].values()}
xs = [s if s > 0 else 0.0 for s in scales]
# place scale 0 at a pseudo-position on the log axis (a quarter of the smallest positive scale)
pos_min = min(s for s in scales if s > 0); x0 = pos_min / 4
xpos = [x0 if s == 0 else s for s in scales]

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(3.6, 2.7), dpi=200)
ceil = J.get("ceilings_dev200", {})
if "vae_roundtrip@512" in ceil:
    ax.axhline(ceil["vae_roundtrip@512"]["f1"], color="0.35", lw=1, ls=":", label="VAE round trip (decode ceiling)")
if "pid_roundtrip@2048" in ceil:
    ax.axhline(ceil["pid_roundtrip@2048"]["f1"], color="tab:orange", lw=0.8, ls=":", alpha=0.7, label="PiD round trip, 2048")
series = [("vae@512", "VAE decode, 512", "0.1", "-", 1.6, "o"),
          (f"pid_k{a.k}@512", f"vanilla PiD (K={a.k}), 512 view", "tab:orange", "-", 1.6, "s"),
          (f"pid_k{a.k}@2048", f"vanilla PiD (K={a.k}), native 2048", "tab:orange", "--", 1.6, "s"),
          ("pid_k28@512", "vanilla PiD (K=28), 512 view", "tab:orange", "-", 0.7, None),
          ("pid_k28@2048", "vanilla PiD (K=28), native 2048", "tab:orange", "--", 0.7, None),
          (f"cgd_k{a.k}@512", f"CGD (K={a.k}), 512 view", "tab:blue", "-", 1.6, "^"),
          (f"cgd_k{a.k}@2048", f"CGD (K={a.k}), native 2048", "tab:blue", "--", 1.6, "^")]
for key, label, color, ls, lw, mk in series:
    ys = [per[s].get(f"{key}_f1") for s in scales]
    if all(y is None for y in ys):
        continue
    ax.plot([x for x, y in zip(xpos, ys) if y is not None], [y for y in ys if y is not None], color=color, ls=ls, lw=lw, marker=mk, ms=3, label=label, alpha=1.0 if lw > 1 else 0.6)
ax.set_xscale("log"); ax.set_xticks(xpos); ax.set_xticklabels([("0" if s == 0 else f"{s:g}") for s in scales])
ax.axvline(1.0, color="0.8", lw=0.6); ax.text(1.0, ax.get_ylim()[1], " released", fontsize=6, color="0.5", va="top")
ax.set_xlabel("control strength at generation (condition scale)"); ax.set_ylabel("canny F1 (pixel space)")
ax.legend(fontsize=6, frameon=False, loc="lower right")
fig.tight_layout(); Path(a.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.out); fig.savefig(str(Path(a.out).with_suffix(".png")))
print("wrote", a.out)
