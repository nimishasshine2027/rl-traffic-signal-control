"""RL agents: tabular Q-learning (own implementation) + wrappers to use DQN / PPO (Stable-Baselines3) as controllers."""
import os, numpy as np

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


class QLearningAgent:
    name = "qlearn"
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95, eps=1.0, eps_min=0.05, eps_decay=0.97, seed=0):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha, self.gamma, self.eps, self.eps_min, self.eps_decay = alpha, gamma, eps, eps_min, eps_decay
        self.rng = np.random.default_rng(seed)
        self.n_actions = n_actions

    def choose(self, s, greedy=False):
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.integers(self.n_actions))
        q = self.Q[s]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))      # random tie-break

    def update(self, s, a, r, s2, done):
        target = r + (0 if done else self.gamma * self.Q[s2].max())
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def decay(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def save(self, tag):
        np.save(os.path.join(MODELS, f"qlearn_{tag}.npy"), self.Q)

    @classmethod
    def load(cls, tag, n_states, n_actions):
        a = cls(n_states, n_actions); a.Q = np.load(os.path.join(MODELS, f"qlearn_{tag}.npy")); a.eps = 0.0
        return a

    # controller interface (evaluation)
    def reset(self): pass
    def act(self, env): return self.choose(env.discrete_state(), greedy=True)


class SB3Controller:
    """Wrap a trained SB3 model (DQN or PPO) so it can be used like the baselines."""
    def __init__(self, algo, tag):
        from stable_baselines3 import DQN, PPO
        self.name = algo
        self.model = {"dqn": DQN, "ppo": PPO}[algo].load(os.path.join(MODELS, f"{algo}_{tag}"), device="cpu")
    def reset(self): pass
    def act(self, env):
        a, _ = self.model.predict(env.last_obs, deterministic=True)
        return int(a)
