"""Baseline controllers (all use the SAME env, so the comparison is fair)."""
import numpy as np


class FixedTime:
    """Cycle through the 4 phases with fixed green durations (s)."""
    name = "fixed"
    def __init__(self, env, greens=(30, 10, 30, 10)): self.g = greens
    def reset(self): pass
    def act(self, env):
        return (env.cur + 1) % env.n_phases if env.green_time >= self.g[env.cur] else env.cur


class Actuated:
    """Detector-based actuated control (emulates stop-line induction loops, 40 m zone):
    extend green while a vehicle is detected near the stop line (up to max-green),
    otherwise gap-out to the next phase that has a vehicle waiting/arriving."""
    name = "actuated"
    def __init__(self, env, max_green=35, zone=40.0): self.mg, self.zone = max_green, zone
    def reset(self): pass
    def _demand(self, env, p):
        c = env.conn
        for l in env.phase_lanes[p]:
            L = env.lane_len
            for v in c.lane.getLastStepVehicleIDs(l):
                if c.vehicle.getLanePosition(v) > L - self.zone: return True
        return False
    def act(self, env):
        cur = env.cur
        if env.green_time < env.min_green: return cur
        if env.green_time < self.mg and self._demand(env, cur): return cur
        for k in range(1, env.n_phases + 1):
            p = (cur + k) % env.n_phases
            if self._demand(env, p): return p
        return cur


class MaxPressure:
    """Max-Pressure (Varaiya 2013): pick the phase with the largest pressure."""
    name = "maxpressure"
    def __init__(self, env): pass
    def reset(self): pass
    def act(self, env):
        pr = [env.phase_pressure(p) for p in range(env.n_phases)]
        best = int(np.argmax(pr))
        return env.cur if pr[env.cur] >= pr[best] else best
