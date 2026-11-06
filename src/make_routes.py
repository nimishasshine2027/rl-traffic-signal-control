"""Step 2: Generate demand files (low / medium / high / varying) as seeded Poisson arrivals."""
import os, random, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "sumo_files")
APPROACHES = ["N", "S", "E", "W"]
OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
LEFT = {"N": "E", "E": "S", "S": "W", "W": "N"}   # driving south from N, left turn goes east
RIGHT = {v: k for k, v in LEFT.items()}

# vehicles/hour per approach, split 60% straight / 20% left / 20% right
LEVELS = {"low": 250, "medium": 500, "high": 750}

def rate_at(profile, t, horizon):
    if profile in LEVELS:
        return LEVELS[profile]
    x = t / horizon          # time-varying: low -> rush peak -> dip -> second peak -> low
    if x < 0.2: return 250
    if x < 0.5: return 750
    if x < 0.7: return 350
    if x < 0.85: return 600
    return 250

def generate(profile, seed, horizon=3600, path=None):
    rng = random.Random(seed)
    veh = []
    for a in APPROACHES:
        t = 0.0
        while True:
            r = rate_at(profile, t, horizon) / 3600.0
            t += rng.expovariate(r)
            if t >= horizon: break
            u = rng.random()
            dest = OPP[a] if u < 0.6 else (LEFT[a] if u < 0.8 else RIGHT[a])
            veh.append((t, a, dest))
    veh.sort()
    path = path or f"{OUT}/{profile}_s{seed}.rou.xml"
    with open(path, "w") as f:
        f.write('<routes>\n<vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="13.89"/>\n')
        for i, (t, a, d) in enumerate(veh):
            f.write(f'<vehicle id="v{i}" type="car" depart="{t:.2f}" departLane="best" departSpeed="max">'
                    f'<route edges="{a}C C{d}"/></vehicle>\n')
        f.write('</routes>\n')
    return path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for p in ["low", "medium", "high", "varying"]:
        for s in range(a.seeds):
            generate(p, s)
    print("routes done")
