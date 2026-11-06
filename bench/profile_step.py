"""Profile one training step (batch 512, with aux) and one snapshot render, printing JSON."""
import json, sys, time
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn.functional as F
from badapple import load_targets, load_aux, logits, frame_losses, display_frames, N, D, FPS

device = torch.device("mps")
logt = load_targets(device)
aux = load_aux(device)
n = logt.shape[0]
torch.manual_seed(0)
x = torch.nn.Parameter(torch.randn(n, N, D, device=device) * 0.1)
wq = torch.nn.Parameter(torch.empty(D, D, device=device).normal_() * 0.02)
wk = torch.nn.Parameter(torch.empty(D, D, device=device).normal_() * 0.02)
opt = torch.optim.Adam([x, wq, wk], lr=0.1)
B = 512
idx = torch.randint(0, n, (B,), device=device)

def step():
    loss = F.mse_loss(logits(x[idx], wq, wk), logt[idx]) + 0.2 * F.mse_loss(x[idx], aux[idx])
    opt.zero_grad(); loss.backward(); opt.step()

for _ in range(5): step()  # warmup
torch.mps.synchronize()
t = {"fwd_bwd_adam_total": []}
t0 = time.perf_counter()
for _ in range(30): step()
torch.mps.synchronize()
t["step_total_s"] = (time.perf_counter() - t0) / 30

# component timing: fwd+bwd without adam
def fb():
    loss = F.mse_loss(logits(x[idx], wq, wk), logt[idx]) + 0.2 * F.mse_loss(x[idx], aux[idx])
    opt.zero_grad(); loss.backward()
for _ in range(5): fb()
torch.mps.synchronize(); t0 = time.perf_counter()
for _ in range(30): fb()
torch.mps.synchronize(); t["fwd_bwd_s"] = (time.perf_counter() - t0) / 30
for _ in range(5): opt.step()
torch.mps.synchronize(); t0 = time.perf_counter()
for _ in range(30): opt.step()
torch.mps.synchronize(); t["adam_s"] = (time.perf_counter() - t0) / 30

# gather-only cost (x[idx] + logt[idx] + aux[idx])
def gather():
    a = x[idx]; b = logt[idx]; c = aux[idx]; return a.sum() + b.sum() + c.sum()
for _ in range(5): gather()
torch.mps.synchronize(); t0 = time.perf_counter()
for _ in range(30): gather()
torch.mps.synchronize(); t["gather_s"] = (time.perf_counter() - t0) / 30

# snapshot display cost (current CPU pipeline), 64 frames sample
ckpt = torch.load("out/final_v1.pt", map_location=device, weights_only=True)
with torch.no_grad():
    s = logits(ckpt["x"][:64], ckpt["wq"], ckpt["wk"]).softmax(-1).cpu().numpy()
t0 = time.perf_counter()
for f in display_frames(s): pass
t["display_cpu_s_per_frame"] = (time.perf_counter() - t0) / 64
print(json.dumps(t, indent=1))
