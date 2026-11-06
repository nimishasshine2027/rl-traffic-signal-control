"""Step 5: evaluate controllers on unseen demand seeds (100+) and save a CSV.
   python src/evaluate.py                                   -> main comparison (all profiles)
   python src/evaluate.py --noise 0.3 --out results/eval_noise03.csv     (sensor-noise robustness)
   python src/evaluate.py --tag medium --profiles high varying --out results/eval_shift.csv  (demand shift)"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from env import TrafficEnv
from controllers import FixedTime, Actuated, MaxPressure
from agents import QLearningAgent, SB3Controller, MODELS

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--controllers", nargs="+", default=["fixed", "actuated", "maxpressure", "qlearn", "dqn", "ppo"])
ap.add_argument("--profiles", nargs="+", default=["low", "medium", "high", "varying"])
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--noise", type=float, default=0.0)
ap.add_argument("--tag", default="mixed")
ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "eval_main.csv"))
a = ap.parse_args()

env = TrafficEnv(profile="medium", obs_noise=a.noise)

def build(name):
    if name == "fixed": return FixedTime(env)
    if name == "actuated": return Actuated(env)
    if name == "maxpressure": return MaxPressure(env)
    if name == "qlearn":
        if not os.path.exists(os.path.join(MODELS, f"qlearn_{a.tag}.npy")): return None
        return QLearningAgent.load(a.tag, env.n_discrete, env.n_phases)
    if not os.path.exists(os.path.join(MODELS, f"{name}_{a.tag}.zip")): return None
    return SB3Controller(name, a.tag)

rows = []
for cname in a.controllers:
    ctrl = build(cname)
    if ctrl is None:
        print(f"(skip {cname}: no trained model with tag '{a.tag}')"); continue
    for prof in a.profiles:
        for s in range(100, 100 + a.seeds):
            env.reset(options={"profile": prof, "seed": s}); done = False
            while not done:
                _, _, _, done, _ = env.step(ctrl.act(env))
            m = env.episode_metrics(); m.update(controller=cname, noise=a.noise, tag=a.tag)
            rows.append(m)
        print(f"{cname:12s} {prof:8s} done")
df = pd.DataFrame(rows)
df.to_csv(a.out, index=False)
summ = df.groupby(["profile", "controller"])[["avg_wait", "avg_travel", "avg_queue", "throughput", "co2_g"]].mean().round(1)
print(summ.to_string())
print("saved", a.out)
env.close()
