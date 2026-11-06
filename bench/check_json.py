"""Check two loss-trajectory JSON files agree. Usage: check_json.py <expected> <actual> <rtol> [final]"""
import json
import sys

a = json.load(open(sys.argv[1]))["losses"]
b = json.load(open(sys.argv[2]))["losses"]
rtol = float(sys.argv[3])
if len(sys.argv) > 4 and sys.argv[4] == "final":
    a, b = [a[-1]], [b[-1]]
assert len(a) == len(b), f"length {len(a)} vs {len(b)}"
for i, (x, y) in enumerate(zip(a, b)):
    assert abs(x - y) <= rtol * max(abs(x), abs(y), 1e-6), f"step {i}: {x} vs {y} (rtol {rtol})"
print("ok")
