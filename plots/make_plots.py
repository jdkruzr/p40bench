#!/usr/bin/env python3
"""P40 bench visualization set. Design per dataviz skill: reference palette
(validated), thin marks, hairline solid grid, text tokens, legends + selective
direct labels, light surface."""
import json, csv, os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

# ---- palette / tokens (light mode) ----
SURF   = "#fcfcfb"
INK    = "#0b0b0b"
INK2   = "#52514e"
MUTED  = "#898781"
GRID   = "#e1e0d9"
AXIS   = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.grid": True, "axes.grid.axis": "y",
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "legend.frameon": False,
})
LINE = dict(lw=2, solid_capstyle="round", marker="o", ms=7,
            markeredgecolor=SURF, markeredgewidth=2)

def styled(ax):
    ax.tick_params(length=0)
    ax.set_axisbelow(True)

def title(fig, t, sub):
    fig.text(0.06, 0.955, t, fontsize=13, fontweight="bold", color=INK, ha="left")
    fig.text(0.06, 0.905, sub, fontsize=9.5, color=INK2, ha="left")

def foot(fig, s):
    fig.text(0.06, 0.015, s, fontsize=8, color=MUTED, ha="left")

OUT = f"{REPO}/plots"
os.makedirs(OUT, exist_ok=True)

# ---- data ----
rows = [json.loads(l) for l in open(f"{REPO}/bench_results/matrix.jsonl")]
def cell(w, topo, mtp, kv, d):
    for r in rows:
        if (r["weights"], r["topo"], r["mtp"], r["kv"], r["target_depth"]) == (w, topo, mtp, kv, d):
            return r["timings"]["predicted_per_second"]
    return None
DEPTHS = [32768, 131072, 261120]
DLAB   = ["32K", "128K", "256K"]

ceres = [(json.loads(l)["n_depth"], json.loads(l)["avg_ts"])
         for l in open(f"{REPO}/ceres_reference/decode_q6_xl.jsonl")
         if json.loads(l)["n_gen"] > 0]
ceres = sorted(set(ceres))

ceres_mult = sorted((int(r["depth"]), float(r["mult"]))
                    for r in csv.DictReader(open(f"{REPO}/ceres_reference/mult_depth.csv"))
                    if r["content"] == "code")

# =====================================================================
# 01 — topology: splitting thinner, not farther
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
fig.subplots_adjust(top=0.82, bottom=0.16, left=0.07, right=0.97, wspace=0.08)
for ax, mtp, panel in zip(axes, ["on", "off"], ["MTP speculative decoding ON", "MTP OFF"]):
    q  = [cell("XL", "quad",   mtp, "q8_0", d) for d in DEPTHS]
    s2 = [cell("XL", "same2",  mtp, "q8_0", d) for d in DEPTHS]
    x2 = cell("XL", "cross2", mtp, "q8_0", 261120)
    ax.plot(DEPTHS, q,  color=BLUE,   label="4 cards", **LINE)
    ax.plot(DEPTHS, s2, color=ORANGE, label="2 cards, same socket", **LINE)
    ax.plot([261120], [x2], color=AQUA, marker="o", ms=7, lw=0,
            markeredgecolor=SURF, markeredgewidth=2, label="2 cards, cross socket")
    ax.annotate("2-card cross-socket:\nidentical to same-socket", xy=(261120, x2),
                xytext=(150000, x2 - 3.4), fontsize=8.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=AXIS, lw=1))
    for d, v in zip(DEPTHS, q):
        ax.annotate(f"{v:.1f}", (d, v), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8.5, color=INK2)
    ax.set_title(panel, fontsize=10, color=INK2, pad=8)
    ax.set_xticks(DEPTHS, DLAB)
    ax.set_xlabel("context depth (tokens)")
    ax.set_ylim(0, 33)
    styled(ax)
axes[0].set_ylabel("decode speed (tokens/s)")
axes[0].legend(loc="lower left", fontsize=9)
title(fig, "Splitting thinner costs; splitting farther is free",
      "Qwen3.6-27B UD-Q6_K_XL on 4× Tesla P40 — decode vs depth by GPU topology, code workload, q8_0 KV")
foot(fig, "llama.cpp b9496 (94a220c), tensor split, batch 1. Cross-socket pair traverses UPI (NUMA0↔NUMA1); measured only at 256K.")
fig.savefig(f"{OUT}/01_topology.png", dpi=150)
plt.close(fig)

# =====================================================================
# 02 — cross-silicon: same engine commit, same weights
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 4.8))
fig.subplots_adjust(top=0.82, bottom=0.17, left=0.07, right=0.97)
cx = [d for d, _ in ceres]; cy = [v for _, v in ceres]
ax.plot(cx, cy, color=BLUE, label="ceres — 2× RTX 5060 Ti (Blackwell, 2024)", **LINE)
py = [cell("XL", "quad", "off", "q8_0", d) for d in DEPTHS]
ax.plot(DEPTHS, py, color=ORANGE, label="4× Tesla P40 (Pascal, 2016)", **LINE)
ax.annotate(f"{cy[-1]:.1f}", (cx[-1], cy[-1]), textcoords="offset points",
            xytext=(0, 10), ha="center", fontsize=8.5, color=INK2)
ax.annotate(f"{py[-1]:.1f}", (DEPTHS[-1], py[-1]), textcoords="offset points",
            xytext=(0, -16), ha="center", fontsize=8.5, color=INK2)
ax.annotate("256K is off the map for 2×16 GB —\nceres's VRAM ceiling for these weights is ~160K",
            xy=(190000, 19.5), fontsize=8.5, color=INK2, ha="center")
ax.set_xticks([0, 32768, 65536, 98304, 131072, 196608, 261120],
              ["0", "32K", "64K", "96K", "128K", "192K", "256K"])
ax.set_xlabel("context depth (tokens)")
ax.set_ylabel("decode speed (tokens/s)")
ax.set_ylim(0, 30)
styled(ax)
ax.legend(loc="lower left", fontsize=9)
title(fig, "Eight years of silicon, one engine commit",
      "Qwen3.6-27B UD-Q6_K_XL, no speculation, q8_0 KV — identical llama.cpp build (b9496 / 94a220c) on both boxes")
foot(fig, "ceres: llama-bench tg128 at depth (June ladder recorded to 64K). P40: live llama-server, code completion, n=200. Methodologies differ ~5–7%.")
fig.savefig(f"{OUT}/02_cross_silicon.png", dpi=150)
plt.close(fig)

# =====================================================================
# 03 — MTP multiplier vs depth across silicon
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 4.6))
fig.subplots_adjust(top=0.82, bottom=0.17, left=0.07, right=0.97)
mx = [d for d, _ in ceres_mult]; my = [v for _, v in ceres_mult]
p40_mult = [cell("XL", "quad", "on", "q8_0", d) / cell("XL", "quad", "off", "q8_0", d)
            for d in DEPTHS]
ax.plot(mx, my, color=BLUE, label="ceres (Blackwell)", **LINE)
ax.plot(DEPTHS, p40_mult, color=ORANGE, label="P40 quad (Pascal)", **LINE)
ax.axhline(1.0, color=AXIS, lw=1)
ax.annotate("1.72× on both boxes at 128K", xy=(131072, 1.724),
            xytext=(120000, 2.02), fontsize=8.5, color=INK2, ha="center",
            arrowprops=dict(arrowstyle="-", color=AXIS, lw=1))
ax.annotate("collapses at 256K\n(draft acceptance 0.89 → 0.57)", xy=(261120, p40_mult[-1]),
            xytext=(228000, 1.45), fontsize=8.5, color=INK2, ha="center",
            arrowprops=dict(arrowstyle="-", color=AXIS, lw=1))
ax.set_xticks([0, 32768, 65536, 98304, 131072, 174000, 261120],
              ["0", "32K", "64K", "96K", "128K", "174K", "256K"])
ax.set_xlabel("context depth (tokens)")
ax.set_ylabel("MTP decode speedup (×, vs no speculation)")
ax.set_ylim(0.9, 2.25)
styled(ax)
ax.legend(loc="upper left", fontsize=9)
title(fig, "Speculative decoding pays more with depth — until it doesn't",
      "MTP speedup on code generation vs context depth: engine property, not silicon property")
foot(fig, "Speedup = decode t/s (spec on) / (spec off), same box, same depth. ceres from June layerD grid; P40 from this run's matrix.")
fig.savefig(f"{OUT}/03_mtp_depth.png", dpi=150)
plt.close(fig)

# =====================================================================
# 04 — fidelity flatline at 128K (dot + CI)
# =====================================================================
ppl = [
    ("UD-Q6_K_XL, q8_0 KV",   5.2076, 0.03226),
    ("UD-Q6_K_XL, f16 KV",    5.2072, 0.03226),
    ("Q8_0, q8_0 KV",         5.2084, 0.03228),
    ("BF16 (reference), q8_0 KV", 5.2090, 0.03229),
]
fig, ax = plt.subplots(figsize=(9, 3.6))
fig.subplots_adjust(top=0.78, bottom=0.18, left=0.27, right=0.95)
ys = range(len(ppl))[::-1]
for y, (label, v, ci) in zip(ys, ppl):
    ax.errorbar(v, y, xerr=ci, color=BLUE, elinewidth=2, capsize=4, capthick=2)
    ax.plot([v], [y], color=BLUE, marker="o", ms=8, markeredgecolor=SURF, markeredgewidth=2)
    ax.annotate(f"{v:.4f}", (v, y), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=8.5, color=INK2)
ax.set_yticks(list(ys), [p[0] for p in ppl])
ax.tick_params(axis="y", colors=INK2)
ax.set_xlim(5.14, 5.28)
ax.set_xlabel("perplexity (wiki.test, 2 × 131,072-token chunks)")
ax.grid(axis="x"); ax.grid(axis="y", visible=False)
styled(ax)
title(fig, "At 128K deep, nothing you did to this model matters",
      "Full-context perplexity, 4 configs: max spread 0.0018 against ±0.032 confidence — a four-way statistical tie")
foot(fig, "4× Tesla P40, 251 GB host RAM (the measurement needs a ~130 GB logits buffer — impossible on 32 GB boxes). BF16 = unquantized.")
fig.savefig(f"{OUT}/04_fidelity.png", dpi=150)
plt.close(fig)

print("wrote 4 plots to", OUT)
