"""Check two rendered-frame npy files agree. Usage: check_npy.py <expected> <actual>"""
import sys

import numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
assert a.shape == b.shape, f"shape {a.shape} vs {b.shape}"
d = np.abs(a.astype(np.int16) - b.astype(np.int16))
mean, mx = d.mean(), d.max()
assert mean < 1.0, f"mean abs diff {mean:.2f}/255"
assert np.percentile(d, 99.9) < 6, f"p99.9 diff {np.percentile(d, 99.9):.1f}/255"
print(f"ok (mean {mean:.2f}, max {mx})")
