"""Run {cmd}, time it, print a JSON metrics object as the last line."""
import json
import shlex
import subprocess
import sys
import time

cmd = " ".join(sys.argv[1:])  # bench.py passes {cmd} pre-split, not as one argv item
t0 = time.perf_counter()
r = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
if r.returncode != 0:
    with open("/tmp/bench_debug.log", "a") as f:
        f.write(f"CMD: {cmd}\nSTDERR: {r.stderr[-2000:]}\n---\n")
    sys.exit(r.returncode)
print(json.dumps({"wall_s": round(time.perf_counter() - t0, 4)}))
