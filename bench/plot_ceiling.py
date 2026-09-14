"""fig:ceiling from results/bench/sweep/condition_scale.json (redesigned 2026-09-14 after the user found the one-panel version unreadable).

Two panels share the x axis (OminiControl condition_scale, log axis; 0 = condition suppressed, drawn at the left; 1 = released setting):
  left  = the matched 512 view,   right = the native 2048 output.
Each panel draws few lines in distinct colours: the VAE decode (black, 512 panel only), vanilla PiD at K = 24 (orange), vanilla PiD at
K = 28 (thin grey), and CGD at K = 24 (blue) when its keys (cgd_k24@512_f1 / @2048_f1) exist in the JSON. Dotted horizontal lines are the
decoder-only ceilings on the same 200 real images: the VAE round trip (512 panel) and the PiD round trip at the panel's resolution.
One shared legend sits below the panels. Usage: python plot_ceiling.py [--out /home/wookiekim/2026-conditional-decoding/figures/ceiling.pdf]
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
ap.add_argument("--k", default="24", help="the truncation drawn as the main generative curve (28 is drawn thin and grey)")
ap.add_argument("--mock-cgd", action="store_true", help="draw a PROJECTED CGD curve (illustrative placeholder, user request 2026-09-14): PiD at K plus a fraction of its gap to the round-trip ceiling that grows with the condition scale (0.3 at scale 0 -> 0.75 at scale >= 1); dashed blue, hollow markers, labelled as projected. Replace with the measured sweep (keys cgd_k<K>@<res>_f1).")
a = ap.parse_args()
J = json.load(open(a.json))
scales = sorted(J["scales"]); per = {m["scale"]: m for m in J["per_scale"].values()}
pos_min = min(s for s in scales if s > 0); x0 = pos_min / 4
xpos = {s: (x0 if s == 0 else s) for s in scales}
ceil = J.get("ceilings_dev200", {})

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), dpi=200, sharey=True)
panels = [("512", "matched 512 view", axes[0]), ("2048", "native 2048 output", axes[1])]
series = [("vae", "VAE decode", "0.1", 1.6, "o"),
          (f"pid_k{a.k}", f"vanilla PiD, $K={a.k}$", "tab:orange", 1.6, "s"),
          ("pid_k28", "vanilla PiD, $K=28$", "0.6", 0.9, None),
          (f"cgd_k{a.k}", f"CGD, $K={a.k}$", "tab:blue", 1.6, "^")]
handles = {}
for res, title, ax in panels:
    if res == "512" and "vae_roundtrip@512" in ceil:
        h = ax.axhline(ceil["vae_roundtrip@512"]["f1"], color="0.35", lw=1, ls=":"); handles.setdefault("VAE round trip (decode ceiling)", h)
    key_rt = f"pid_roundtrip@{res}"
    if key_rt in ceil:
        h = ax.axhline(ceil[key_rt]["f1"], color="tab:orange", lw=1, ls=":"); handles.setdefault("PiD round trip (generative ceiling)", h)
    for key, label, color, lw, mk in series:
        ys = [per[s].get(f"{key}@{res}_f1") for s in scales]
        if all(y is None for y in ys):
            continue
        xs = [xpos[s] for s, y in zip(scales, ys) if y is not None]; ys = [y for y in ys if y is not None]
        (h,) = ax.plot(xs, ys, color=color, lw=lw, marker=mk, ms=3, label=label); handles.setdefault(label, h)
    if a.mock_cgd and key_rt in ceil and all(per[s].get(f"pid_k{a.k}@{res}_f1") is not None for s in scales):
        # PROJECTED placeholder: closes a scale-dependent fraction of PiD's gap to the round-trip ceiling (correction at low scale, preservation at high scale)
        c = ceil[key_rt]["f1"]
        alpha = lambda s: 0.3 + 0.45 * min(s, 1.0)  # noqa: E731
        ys = [per[s][f"pid_k{a.k}@{res}_f1"] + alpha(s) * (c - per[s][f"pid_k{a.k}@{res}_f1"]) for s in scales]
        (h,) = ax.plot([xpos[s] for s in scales], ys, color="tab:blue", lw=1.6, ls="--", marker="^", ms=4, mfc="white")
        handles.setdefault(f"CGD, $K={a.k}$ (projected, not measured)", h)
        ax.text(xpos[scales[-1]], ys[-1] + 0.025, "projected", fontsize=6, color="tab:blue", ha="right")
    ax.set_xscale("log")
    major = [s for s in scales if s in (0, 0.25, 0.5, 1.0, 2.0, 4.0)]
    ax.set_xticks([xpos[s] for s in major]); ax.set_xticklabels([("0" if s == 0 else f"{s:g}") for s in major])
    ax.set_xticks([xpos[s] for s in scales if s not in major], minor=True); ax.tick_params(axis="x", which="minor", length=2)
    ax.axvline(1.0, color="0.85", lw=0.8, zorder=0); ax.text(1.0, 0.205, "released", fontsize=6, color="0.5", ha="center", va="bottom")
    ax.set_title(title, fontsize=8); ax.set_xlabel("condition scale at generation")
    ax.set_ylim(0.2, 1.0)
axes[0].set_ylabel("canny F1 (tolerant)")
order = ["VAE round trip (decode ceiling)", "PiD round trip (generative ceiling)", "VAE decode", f"vanilla PiD, $K={a.k}$", "vanilla PiD, $K=28$", f"CGD, $K={a.k}$", f"CGD, $K={a.k}$ (projected, not measured)"]
labels = [l for l in order if l in handles]
fig.legend([handles[l] for l in labels], labels, loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.12, 1, 1)); Path(a.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.out); fig.savefig(str(Path(a.out).with_suffix(".png")))
print("wrote", a.out)
