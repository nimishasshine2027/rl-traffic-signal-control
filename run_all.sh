#!/usr/bin/env bash
# Full pipeline (Linux/Mac/Git-Bash). ~25-35 min total on a laptop. On Windows run the python lines one by one.
set -e
python src/make_network.py
python src/make_routes.py --seeds 20
python src/train.py --algo qlearn --episodes 100
python src/train.py --algo dqn --steps 60000
python src/train.py --algo ppo --steps 60000
python src/evaluate.py                                                  # main comparison
python src/evaluate.py --noise 0.3 --out results/eval_noise03.csv      # robustness: sensor noise
python src/train.py --algo qlearn --episodes 100 --profile medium --tag medium   # demand-shift experiment
python src/train.py --algo dqn --steps 60000 --profile medium --tag medium
python src/train.py --algo ppo --steps 60000 --profile medium --tag medium
python src/evaluate.py --tag medium --profiles high varying --controllers qlearn dqn ppo --out results/eval_shift.csv
python src/plot_results.py
