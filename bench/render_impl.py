"""Snapshot-render benchmark impl. Usage: render_impl.py <base|gpu> <out.npy>.

Renders 256 frames from the v1 checkpoint through the given display pipeline
and saves the RGB frames for output checking.
"""
import sys

sys.path.insert(0, ".")
import numpy as np
import torch

from badapple import display_frames, logits, N

variant, out = sys.argv[1], sys.argv[2]
device = torch.device("mps")
ckpt = torch.load("out/final_v1.pt", map_location=device, weights_only=True)
with torch.no_grad():
    attn = logits(ckpt["x"][:256], ckpt["wq"], ckpt["wk"]).softmax(-1)

if variant == "base":
    imgs = display_frames(attn.float().cpu().numpy())
else:
    import torch.nn.functional as F
    import matplotlib
    CMAP = matplotlib.colormaps["magma"]

    z = (attn - attn.mean(-1, keepdim=True)) / (attn.std(-1, keepdim=True) + 1e-8)
    flat = z.reshape(attn.shape[0], -1)
    srt = flat.sort(dim=1).values
    n1 = flat.shape[1] - 1
    p1, p2 = 0.01 * n1, 0.99 * n1  # match np.percentile's linear interpolation exactly
    lo = (1 - (p1 % 1)) * srt[:, int(p1):int(p1) + 1] + (p1 % 1) * srt[:, int(p1) + 1:int(p1) + 2]
    hi = (1 - (p2 % 1)) * srt[:, int(p2):int(p2) + 1] + (p2 % 1) * srt[:, int(p2) + 1:int(p2) + 2]
    img = ((flat - lo) / (hi - lo + 1e-8)).clamp(0, 1).reshape(attn.shape)
    r = 6  # scipy truncate=4 default -> radius 6
    xs = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    kern = (torch.exp(-xs**2 / (2 * 1.5**2)) / torch.exp(-xs**2 / (2 * 1.5**2)).sum()).view(1, 1, 1, -1)
    im = F.conv2d(F.pad(img[:, None], (r, r, 0, 0), mode="replicate"), kern)
    im = F.conv2d(F.pad(im, (0, 0, r, r), mode="replicate"), kern.transpose(-1, -2))
    lut = torch.from_numpy((matplotlib.colormaps["magma"](np.linspace(0, 1, 256))[..., :3] * 255).astype(np.uint8)).to(device)
    q = (im[:, 0] * 255).round().long().clamp(0, 255)
    imgs = lut[q].cpu().numpy()

with open(out, "wb") as f:  # np.save(str) appends .npy; bench passes a suffix-less path
    np.save(f, imgs)
