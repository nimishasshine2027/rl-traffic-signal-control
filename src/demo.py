"""Step 7: LIVE DEMO. Runs one episode and prints KPIs; with --gui opens the SUMO window so you can SEE the signals.
   python src/demo.py --controller fixed --profile high --gui
   python src/demo.py --controller dqn   --profile high --gui
   python src/demo.py --controller maxpressure ppo qlearn fixed --profile varying       (headless table)"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from env import TrafficEnv
from controllers import FixedTime, Actuated, MaxPressure
from agents import QLearningAgent, SB3Controller

ap = argparse.ArgumentParser()
ap.add_argument("--controller", nargs="+", default=["fixed", "maxpressure"])
ap.add_argument("--profile", default="varying")
ap.add_argument("--seed", type=int, default=100)
ap.add_argument("--tag", default="mixed")
ap.add_argument("--gui", action="store_true")
ap.add_argument("--delay", type=int, default=60, help="ms per sim step in GUI (higher = slower)")
a = ap.parse_args()

rows = []
for name in a.controller:
    env = TrafficEnv(profile=a.profile, gui=a.gui, gui_delay=a.delay)
    ctrl = {"fixed": lambda: FixedTime(env), "actuated": lambda: Actuated(env), "maxpressure": lambda: MaxPressure(env),
            "qlearn": lambda: QLearningAgent.load(a.tag, env.n_discrete, env.n_phases),
            "dqn": lambda: SB3Controller("dqn", a.tag), "ppo": lambda: SB3Controller("ppo", a.tag)}[name]()
    env.reset(options={"profile": a.profile, "seed": a.seed}); done = False
    print(f"\n=== {name} | demand={a.profile} | seed={a.seed} ===")
    while not done:
        _, _, _, done, _ = env.step(ctrl.act(env))
        if env.t % 600 < env.delta + env.yellow:
            print(f"  t={env.t:4d}s  phase={env.cur}  queue now={int(env.lane_queues().sum()):3d}")
    m = env.episode_metrics(); m["controller"] = name; rows.append(m); env.close()
df = pd.DataFrame(rows).set_index("controller")[["avg_wait", "avg_travel", "avg_queue", "throughput", "co2_g"]].round(1)
print("\n", df.to_string())
