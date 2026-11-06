"""Step 4: train an RL agent.
   python src/train.py --algo qlearn --episodes 60
   python src/train.py --algo dqn --steps 60000
   python src/train.py --algo ppo --steps 60000
Extra experiments:  --profile medium --tag medium   (train on one demand level only, for the demand-shift test)
                    --reward queue|pressure|wait     (reward-design ablation, use --tag to name it)
                    --noise 0.2                      (train with noisy sensors)"""
import os, sys, argparse, csv, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from env import TrafficEnv
from agents import QLearningAgent, MODELS

ap = argparse.ArgumentParser()
ap.add_argument("--algo", choices=["qlearn", "dqn", "ppo"], required=True)
ap.add_argument("--profile", default="mixed", help="low|medium|high|varying|mixed")
ap.add_argument("--reward", default="wait")
ap.add_argument("--noise", type=float, default=0.0)
ap.add_argument("--episodes", type=int, default=60, help="qlearn only")
ap.add_argument("--steps", type=int, default=60000, help="dqn/ppo only")
ap.add_argument("--tag", default="mixed")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results"); os.makedirs(RES, exist_ok=True); os.makedirs(MODELS, exist_ok=True)
env = TrafficEnv(profile=a.profile, seeds=range(0, 20), reward=a.reward, obs_noise=a.noise)   # training seeds 0-19
t0 = time.time()

if a.algo == "qlearn":
    agent = QLearningAgent(env.n_discrete, env.n_phases, seed=a.seed,
                           eps_decay=0.05 ** (1 / max(1, int(0.7 * a.episodes))))   # epsilon hits 0.05 at 70% of training
    log = open(os.path.join(RES, f"train_qlearn_{a.tag}.csv"), "w", newline="")
    w = csv.writer(log); w.writerow(["episode", "reward", "avg_wait", "epsilon"])
    for ep in range(a.episodes):
        env.reset(seed=a.seed + ep)
        s, done, R = env.discrete_state(), False, 0.0
        while not done:
            act = agent.choose(s)
            _, r, _, done, _ = env.step(act)
            s2 = env.discrete_state()
            agent.update(s, act, r, s2, done)
            s, R = s2, R + r
        m = env.episode_metrics()
        agent.decay()
        w.writerow([ep, R, m["avg_wait"], agent.eps]); log.flush(); agent.save(a.tag)
        print(f"ep {ep+1}/{a.episodes} [{m['profile']}] return={R:8.1f} avg_wait={m['avg_wait']:.1f}s eps={agent.eps:.2f}  ({time.time()-t0:.0f}s)")
    agent.save(a.tag)
else:
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.monitor import Monitor
    menv = Monitor(env, os.path.join(RES, f"train_{a.algo}_{a.tag}"))
    if a.algo == "dqn":
        model = DQN("MlpPolicy", menv, learning_rate=1e-3, buffer_size=50_000, learning_starts=1000,
                    batch_size=64, gamma=0.95, train_freq=4, target_update_interval=500,
                    exploration_fraction=0.4, exploration_final_eps=0.05,
                    policy_kwargs=dict(net_arch=[64, 64]), seed=a.seed, device="cpu", verbose=0)
    else:
        model = PPO("MlpPolicy", menv, learning_rate=3e-4, n_steps=1440, batch_size=144, n_epochs=10,
                    gamma=0.95, ent_coef=0.01, policy_kwargs=dict(net_arch=[64, 64]), seed=a.seed,
                    device="cpu", verbose=0)
    model.learn(total_timesteps=a.steps, progress_bar=False)
    model.save(os.path.join(MODELS, f"{a.algo}_{a.tag}"))
print(f"saved model '{a.algo}_{a.tag}'  total {time.time()-t0:.0f}s")
env.close()
