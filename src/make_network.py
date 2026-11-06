"""Step 1: Build a 4-way intersection (3 lanes per approach, 300 m arms) with netconvert."""
import os, subprocess
import sumo

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "sumo_files")
os.makedirs(OUT, exist_ok=True)
NETCONVERT = os.path.join(sumo.SUMO_HOME, "bin", "netconvert")

nodes = """<nodes>
  <node id="C" x="0" y="0" type="traffic_light"/>
  <node id="N" x="0" y="300"/>
  <node id="S" x="0" y="-300"/>
  <node id="E" x="300" y="0"/>
  <node id="W" x="-300" y="0"/>
</nodes>"""
edges_l = ['<edges>']
for a in "NSEW":
    edges_l.append(f'  <edge id="{a}C" from="{a}" to="C" numLanes="3" speed="13.89"/>')
    edges_l.append(f'  <edge id="C{a}" from="C" to="{a}" numLanes="3" speed="13.89"/>')
edges_l.append('</edges>')
open(f"{OUT}/inter.nod.xml", "w").write(nodes)
open(f"{OUT}/inter.edg.xml", "w").write("\n".join(edges_l))

# explicit lane use: lane0 = right+straight, lane1 = straight, lane2 = LEFT only (no U-turns)
OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
LEFT = {"N": "E", "E": "S", "S": "W", "W": "N"}
RIGHT = {v: k for k, v in LEFT.items()}
con = ["<connections>"]
for a in "NSEW":
    con.append(f'  <connection from="{a}C" to="C{RIGHT[a]}" fromLane="0" toLane="0"/>')
    con.append(f'  <connection from="{a}C" to="C{OPP[a]}" fromLane="0" toLane="0"/>')
    con.append(f'  <connection from="{a}C" to="C{OPP[a]}" fromLane="1" toLane="1"/>')
    con.append(f'  <connection from="{a}C" to="C{LEFT[a]}" fromLane="2" toLane="2"/>')
con.append("</connections>")
open(f"{OUT}/inter.con.xml", "w").write("\n".join(con))

cmd = [NETCONVERT, "-x", f"{OUT}/inter.con.xml", "--no-turnarounds", "true", "-n", f"{OUT}/inter.nod.xml", "-e", f"{OUT}/inter.edg.xml",
       "-o", f"{OUT}/inter.net.xml", "--tls.default-type", "static", "--no-warnings", "true"]
subprocess.check_call(cmd)
print("Network written to", f"{OUT}/inter.net.xml")
