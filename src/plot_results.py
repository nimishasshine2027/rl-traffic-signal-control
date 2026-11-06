"""Step 6: make the figures for the report / presentation (saved in results/)."""
import os, sys, glob, argparse
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
ORDER = ["fixed", "actuated", "maxpressure", "qlearn", "dqn", "ppo"]
COL = {"fixed": "#9e9e9e", "actuated": "#bdbdbd", "maxpressure": "#607d8b", "qlearn": "#ff9800", "dqn": "#1976d2", "ppo": "#2e7d32"}
METRICS = [("avg_wait", "Avg waiting time (s)  ↓"), ("avg_travel", "Avg travel time (s)  ↓"), ("avg_queue", "Avg queue length (veh)  ↓"),
           ("throughput", "Throughput (veh / h)  ↑"), ("co2_g", "CO₂ emitted (g)  ↓")]

def grouped(df, group, fname, title):
    cs = [c for c in ORDER if c in df.controller.unique()]
    groups = list(dict.fromkeys(df[group]))
    fig, axs = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 4))
    for ax, (m, lab) in zip(axs, METRICS):
        w = 0.8 / len(cs)
        for i, c in enumerate(cs):
            g = df[df.controller == c].groupby(group)[m]
            mu = [g.mean().get(k, np.nan) for k in groups]; sd = [g.std().get(k, 0) for k in groups]
            ax.bar(np.arange(len(groups)) + i * w, mu, w, yerr=sd, label=c, color=COL[c], capsize=2)
        ax.set_xticks(np.arange(len(groups)) + 0.4 - w / 2); ax.set_xticklabels(groups); ax.set_title(lab, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].legend(fontsize=8)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(os.path.join(R, fname), dpi=140); plt.close(fig)
    print("wrote", fname)

f = os.path.join(R, "eval_main.csv")
if os.path.exists(f): grouped(pd.read_csv(f), "profile", "fig_main_comparison.png", "Controller comparison per demand level (mean ± sd over unseen seeds)")
for name, fn, t in [("eval_shift.csv", "fig_demand_shift.png", "Demand shift: agents trained on MEDIUM only, tested on unseen demand"),]:
    if os.path.exists(os.path.join(R, name)): grouped(pd.read_csv(os.path.join(R, name)), "profile", fn, t)
# noise robustness: concat all eval_noise*.csv + main (noise 0) on 'varying'
parts = [pd.read_csv(p) for p in glob.glob(os.path.join(R, "eval_noise*.csv"))]
if parts and os.path.exists(f):
    d = pd.concat([pd.read_csv(f)] + parts); d = d[d.profile == "varying"]
    d["noise"] = d["noise"].astype(str)
    grouped(d, "noise", "fig_sensor_noise.png", "Sensor-noise robustness (varying demand); x-axis = noise level σ")

# training curves
fig, ax = plt.subplots(1, 1, figsize=(7, 4))
for algo in ["qlearn", "dqn", "ppo"]:
    p = os.path.join(R, f"train_{algo}_mixed" + (".csv" if algo == "qlearn" else ".monitor.csv"))
    if not os.path.exists(p): continue
    d = pd.read_csv(p, skiprows=0 if algo == "qlearn" else 1)
    y = d["reward"] if algo == "qlearn" else d["r"]
    ax.plot(y.rolling(8, min_periods=1).mean().values, label=algo, color=COL[algo])
ax.set_xlabel("training episode"); ax.set_ylabel("episode return (smoothed)"); ax.legend(); ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Learning curves"); fig.tight_layout(); fig.savefig(os.path.join(R, "fig_learning_curves.png"), dpi=140)
print("wrote fig_learning_curves.png")
