# RL-Based Adaptive Traffic Signal Control (SUMO)

Single 4-way intersection (3 lanes/approach, protected left turns), 4 demand profiles
(low 250, medium 500, high 750 veh/h/approach, time-varying), 3 RL agents vs 3 baselines.

| file | purpose |
|---|---|
| src/make_network.py | builds the intersection with netconvert |
| src/make_routes.py  | seeded Poisson demand (low/medium/high/varying) |
| src/env.py          | the MDP: state, action, reward, KPIs (Gymnasium env over SUMO/TraCI) |
| src/controllers.py  | baselines: fixed-time, actuated, max-pressure |
| src/agents.py       | tabular Q-learning + DQN/PPO wrappers |
| src/train.py        | training (Q-learning, DQN, PPO) + ablation flags |
| src/evaluate.py     | fair evaluation on unseen seeds -> CSV |
| src/plot_results.py | figures |
| src/demo.py         | live demo (--gui shows SUMO window) |

Note: "Retrying in 1 seconds" printed at start of every episode is TraCI waiting for SUMO to open its socket. It is normal.
