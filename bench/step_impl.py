"""Training-step benchmark impl. Usage: step_impl.py <variant> <out.json> <seedfile>.

Runs a fixed number of frame-updates (25600) so variants with different batch
sizes are comparable, and writes the loss trajectory for output checking.
"""
import json
import sys

sys.path.insert(0, ".")
import torch
import torch.nn.functional as F

from badapple import load_targets, load_aux, logits, N, D

variant, out, inp = sys.argv[1], sys.argv[2], sys.argv[3]
seed = int(open(inp).read().strip())
device = torch.device("cpu" if variant.startswith("cpu") else "mps")
torch.manual_seed(seed)
logt = load_targets(device)
aux = load_aux(device)
n = logt.shape[0]
x = torch.nn.Parameter(torch.randn(n, N, D, device=device) * 0.1)
wq = torch.nn.Parameter(torch.empty(D, D, device=device).normal_() * 0.02)
wk = torch.nn.Parameter(torch.empty(D, D, device=device).normal_() * 0.02)
losses = []

if variant in ("base", "bf16", "cpu", "cpu1024"):
    if variant == "bf16":
        logt, aux = logt.bfloat16(), aux.bfloat16()
    steps, B = (25, 1024) if variant == "cpu1024" else (50, 512)
    # same param-group lrs as the real train(): shared Q/K moves 10x slower
    opt = torch.optim.Adam([{"params": [x], "lr": 0.1}, {"params": [wq, wk], "lr": 0.01}])
    for _ in range(steps):
        idx = torch.randint(0, n, (B,), device=device)
        loss = F.mse_loss(logits(x[idx], wq, wk), logt[idx].float()) + 0.2 * F.mse_loss(x[idx], aux[idx].float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

elif variant == "sparse":
    steps, B = 50, 512
    # lazy Adam with catch-up decay: touched rows first decay their moments by the
    # steps elapsed since their last touch, making the update identical to dense Adam
    opt_qk = torch.optim.Adam([wq, wk], lr=0.01)
    m = torch.zeros_like(x)
    v = torch.zeros_like(x)
    last = torch.zeros(n, device=device)
    b1, b2, eps, lr = 0.9, 0.999, 1e-8, 0.1
    for t in range(1, steps + 1):
        idx = torch.randint(0, n, (B,), device=device)
        xi = x.data[idx].requires_grad_(True)
        loss = F.mse_loss(logits(xi, wq, wk), logt[idx]) + 0.2 * F.mse_loss(xi, aux[idx])
        gx, gwq, gwk = torch.autograd.grad(loss, [xi, wq, wk])
        with torch.no_grad():
            # global-t bias correction + catch-up decay == dense Adam exactly
            elapsed = (t - last[idx]).clamp(min=1).float().view(-1, 1, 1)
            m[idx] = m[idx] * b1 ** elapsed + (1 - b1) * gx
            v[idx] = v[idx] * b2 ** elapsed + (1 - b2) * gx * gx
            x.data[idx] = x.data[idx] - lr * (m[idx] / (1 - b1 ** t)) / ((v[idx] / (1 - b2 ** t)).sqrt() + eps)
            last[idx] = t
        wq.grad, wk.grad = gwq, gwk
        opt_qk.step()
        opt_qk.zero_grad(set_to_none=True)
        losses.append(loss.item())

elif variant == "big1024":
    steps, B = 25, 1024
    opt = torch.optim.Adam([x, wq, wk], lr=0.1)
    for _ in range(steps):
        idx = torch.randint(0, n, (B,), device=device)
        loss = F.mse_loss(logits(x[idx], wq, wk), logt[idx]) + 0.2 * F.mse_loss(x[idx], aux[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

else:
    sys.exit(f"unknown variant {variant}")

json.dump({"losses": losses, "updates": 25600}, open(out, "w"))
