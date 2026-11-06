"""
Step 3: SUMO traffic-signal environment (Gymnasium API) = the MDP.

  STATE   : [phase one-hot (4) | elapsed green (1) | lane density (12) | lane queue (12)]
  ACTION  : choose which of the 4 green phases to serve next
              0 = N/S through+right   1 = N/S left
              2 = E/W through+right   3 = E/W left
            (3 s yellow is inserted automatically on change; min-green = 10 s)
  REWARD  : 'wait'  -> reduction in total waiting time on incoming lanes  (default)
            'queue' -> -(total queue) 
            'pressure' -> -(|pressure|) of the intersection
"""
import os, glob, shutil, tempfile, itertools, random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traci
import sumo as _sumo
import xml.etree.ElementTree as ET

from make_routes import generate, OPP, LEFT

HERE = os.path.dirname(os.path.abspath(__file__))
SUMO_DIR = os.path.join(HERE, "..", "sumo_files")
NET = os.path.join(SUMO_DIR, "inter.net.xml")
TLS = "C"
PROFILES = ["low", "medium", "high", "varying"]
_counter = itertools.count()


def _binary(gui):
    name = "sumo-gui" if gui else "sumo"
    p = os.path.join(_sumo.SUMO_HOME, "bin", name + (".exe" if os.name == "nt" else ""))
    return p if os.path.exists(p) else (shutil.which(name) or name)


class TrafficEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, profile="medium", seeds=range(10), sim_len=3600, delta=5, yellow=3,
                 min_green=10, reward="wait", obs_noise=0.0, gui=False, gui_delay=0, tag=""):
        super().__init__()
        self.profiles = PROFILES if profile == "mixed" else [profile]
        self.seeds = list(seeds)
        self.sim_len, self.delta, self.yellow, self.min_green = sim_len, delta, yellow, min_green
        self.reward_type, self.obs_noise, self.gui, self.gui_delay = reward, obs_noise, gui, gui_delay
        self.label = f"env{os.getpid()}_{next(_counter)}{tag}"
        self.conn = None
        self.tmp = tempfile.mkdtemp(prefix="tsc_")
        self._read_network_structure()
        self.n_phases = len(self.green_states)
        self.n_lanes = len(self.in_lanes)
        self.action_space = spaces.Discrete(self.n_phases)
        dim = self.n_phases + 1 + 2 * self.n_lanes
        self.observation_space = spaces.Box(0.0, 1.0, shape=(dim,), dtype=np.float32)
        self.rng = np.random.default_rng(0)

    # ---------- network analysis (done once, using a short throw-away SUMO run) ----------
    def _read_network_structure(self):
        label = self.label + "_probe"
        traci.start([_binary(False), "-n", NET, "--no-step-log", "true", "--no-warnings", "true"], label=label)
        c = traci.getConnection(label)
        links = c.trafficlight.getControlledLinks(TLS)
        self.links = []                              # (idx, approach, turn, in_lane, out_lane)
        for i, group in enumerate(links):
            for (inl, outl, _via) in group:
                a, d = c.lane.getEdgeID(inl)[0], c.lane.getEdgeID(outl)[1]
                turn = "S" if d == OPP[a] else ("L" if d == LEFT[a] else "R")
                if d == a: continue                      # ignore U-turns
                self.links.append((i, a, turn, inl, outl))
        self.n_signals = len(links)
        self.in_lanes = sorted({l[3] for l in self.links})
        self.out_lanes = sorted({l[4] for l in self.links})
        self.lane_len = c.lane.getLength(self.in_lanes[0])
        c.close()
        # four phases: (approaches, left-turn phase?)
        defs = [("NS", False), ("NS", True), ("EW", False), ("EW", True)]
        self.green_states, self.phase_links, self.phase_lanes = [], [], []
        for appr, left in defs:
            st = ["r"] * self.n_signals
            pl = []
            for (i, a, turn, inl, outl) in self.links:
                if a in appr and ((turn == "L") if left else (turn != "L")):
                    st[i] = "G"
                    pl.append((inl, outl))
            self.green_states.append("".join(st))
            self.phase_links.append(pl)
            self.phase_lanes.append(sorted({x[0] for x in pl}))
        self.approach_lanes = {a: [l for l in self.in_lanes if l[0] == a] for a in "NSEW"}
        self.edges = sorted({l.rsplit("_", 1)[0] for l in self.in_lanes + self.out_lanes})

    # ---------- gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._close_sim()
        options = options or {}
        profile = options.get("profile") or self.np_random.choice(self.profiles)
        s = options.get("seed")
        s = int(self.np_random.choice(self.seeds)) if s is None else s
        route = os.path.join(SUMO_DIR, f"{profile}_s{s}.rou.xml")
        if not os.path.exists(route):
            generate(profile, s)
        self.profile, self.route_seed = profile, s
        self.tripfile = os.path.join(self.tmp, f"trip_{self.label}.xml")
        cmd = [_binary(self.gui), "-n", NET, "-r", route, "--seed", str(s),
               "--no-step-log", "true", "--no-warnings", "true", "--time-to-teleport", "-1",
               "--tripinfo-output", self.tripfile, "--waiting-time-memory", "1000"]
        if self.gui:
            cmd += ["--start", "true", "--quit-on-end", "true", "--delay", str(self.gui_delay)]
        traci.start(cmd, label=self.label)
        self.conn = traci.getConnection(self.label)
        self.cur = 0
        self.green_time = 0
        self.conn.trafficlight.setRedYellowGreenState(TLS, self.green_states[0])
        self.t = 0
        self.m = dict(queue_sum=0.0, queue_max=0.0, co2=0.0, fuel=0.0, steps=0, arrived=0, switches=0)
        self.prev_wait = self._total_wait()
        return self._obs(), {}

    def step(self, action):
        action = int(action)
        if action != self.cur and self.green_time >= self.min_green:
            old = self.green_states[self.cur]
            new = self.green_states[action]
            ystate = "".join("y" if o in "Gg" and n == "r" else o for o, n in zip(old, new))
            self.conn.trafficlight.setRedYellowGreenState(TLS, ystate)
            self._sim(self.yellow)
            self.conn.trafficlight.setRedYellowGreenState(TLS, new)
            self.cur, self.green_time = action, 0
            self.m["switches"] += 1
        self._sim(self.delta)
        self.green_time += self.delta
        r = self._reward()
        done = self.t >= self.sim_len
        return self._obs(), r, False, done, {}

    def close(self):
        self._close_sim()

    # ---------- helpers ----------
    def _sim(self, seconds):
        c = self.conn
        for _ in range(seconds):
            c.simulationStep()
            self.t += 1
            q = sum(c.lane.getLastStepHaltingNumber(l) for l in self.in_lanes)
            self.m["queue_sum"] += q
            self.m["queue_max"] = max(self.m["queue_max"], q)
            self.m["arrived"] += c.simulation.getArrivedNumber()
            for e in self.edges:
                self.m["co2"] += c.edge.getCO2Emission(e) / 1000.0      # mg/s -> g
                self.m["fuel"] += c.edge.getFuelConsumption(e) / 1000.0  # mg/s -> g
            self.m["steps"] += 1

    def lane_queues(self):
        return np.array([self.conn.lane.getLastStepHaltingNumber(l) for l in self.in_lanes], dtype=np.float32)

    def lane_counts(self):
        return np.array([self.conn.lane.getLastStepVehicleNumber(l) for l in self.in_lanes], dtype=np.float32)

    def _total_wait(self):
        return sum(self.conn.lane.getWaitingTime(l) for l in self.in_lanes)

    def _reward(self):
        if self.reward_type == "wait":
            w = self._total_wait()
            r = (self.prev_wait - w) / 100.0
            self.prev_wait = w
            return float(r)
        if self.reward_type == "queue":
            return -float(self.lane_queues().sum()) / 10.0
        if self.reward_type == "pressure":
            return -abs(self.pressure_of_all()) / 10.0
        raise ValueError(self.reward_type)

    def pressure_of_all(self):
        q_in = sum(self.conn.lane.getLastStepVehicleNumber(l) for l in self.in_lanes)
        q_out = sum(self.conn.lane.getLastStepVehicleNumber(l) for l in self.out_lanes)
        return q_in - q_out

    def phase_pressure(self, p):
        """Max-Pressure: sum over served movements of (incoming queue - outgoing queue)."""
        c = self.conn
        return sum(c.lane.getLastStepVehicleNumber(i) - c.lane.getLastStepVehicleNumber(o)
                   for i, o in self.phase_links[p])

    def _sensor(self):
        """lane queue & count as the agent SEES them (optionally noisy)."""
        q, n = self.lane_queues(), self.lane_counts()
        if self.obs_noise > 0:
            q = np.clip(q + self.np_random.normal(0, self.obs_noise * 5, q.shape), 0, None)
            n = np.clip(n + self.np_random.normal(0, self.obs_noise * 5, n.shape), 0, None)
        return q, n

    def _obs(self):
        q, n = self._sensor()
        self._last_q = q
        cap = self.lane_len / 7.5
        ph = np.zeros(self.n_phases, dtype=np.float32); ph[self.cur] = 1
        el = min(self.green_time, 60) / 60.0
        o = np.concatenate([ph, [el], np.clip(n / cap, 0, 1), np.clip(q / cap, 0, 1)]).astype(np.float32)
        self.last_obs = o
        return o

    def discrete_state(self):
        """compact state for tabular Q-learning: phase + 4 binned approach queues."""
        q = self._last_q
        idx = {l: k for k, l in enumerate(self.in_lanes)}
        s = [self.cur]
        for a in "NSEW":
            tot = sum(q[idx[l]] for l in self.approach_lanes[a])
            s.append(int(np.digitize(tot, [1, 5, 12])))        # 0, 1-4, 5-11, 12+
        code = 0
        for v, base in zip(s, [self.n_phases, 4, 4, 4, 4]):
            code = code * base + v
        return code

    @property
    def n_discrete(self):
        return self.n_phases * 4 ** 4

    def episode_metrics(self):
        """call at end of episode (before reset/close). Returns dict of KPIs."""
        self.conn.close(); self.conn = None
        wt, dur, tl = [], [], []
        if os.path.exists(self.tripfile):
            for ti in ET.parse(self.tripfile).getroot().iter("tripinfo"):
                wt.append(float(ti.get("waitingTime"))); dur.append(float(ti.get("duration")))
                tl.append(float(ti.get("timeLoss")))
        m = self.m
        return dict(profile=self.profile, seed=self.route_seed,
                    avg_wait=np.mean(wt) if wt else np.nan, avg_travel=np.mean(dur) if dur else np.nan,
                    avg_queue=m["queue_sum"] / max(m["steps"], 1), max_queue=m["queue_max"],
                    throughput=len(wt), co2_g=m["co2"], fuel_g=m["fuel"], switches=m["switches"])

    def _close_sim(self):
        if self.conn is not None:
            try: self.conn.close()
            except Exception: pass
            self.conn = None
