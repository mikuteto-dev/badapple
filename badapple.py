"""Train one attention head (its input embeddings, Q and K) to reproduce Bad Apple
frames in its 256x256 attention matrix, and play checkpoints back as video.
Unlike the frozen-GPT2 variant, the whole display is learned in one run.
"""

import argparse
import math
import os
import subprocess
import time

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

VIDEO_URL = "https://www.youtube.com/watch?v=FtutLA63Cp8"
VIDEO = "frames/bad_apple.mp4"
AUDIO = "frames/bad_apple_audio.mp3"
TARGETS = "frames/targets"
OUT = "out"
N = 256  # token count == attention matrix side == frame resolution
D = 64   # head dim; logits stay rank-D, the same effective rank that worked for GPT-2 XL
FPS = 15
CMAP = matplotlib.colormaps["magma"]


def download():
    os.makedirs(TARGETS, exist_ok=True)
    if not os.path.exists(VIDEO):
        subprocess.run(
            ["yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
             "--merge-output-format", "mp4", "-o", VIDEO, VIDEO_URL],
            check=True,
        )
    if not os.path.exists(AUDIO):
        subprocess.run(["ffmpeg", "-y", "-i", VIDEO, "-vn", "-acodec", "libmp3lame", "-q:a", "2", AUDIO],
                       check=True, capture_output=True)
    if len(os.listdir(TARGETS)) < 10:
        subprocess.run(["ffmpeg", "-y", "-i", VIDEO, "-vf", f"fps={FPS},scale={N}:{N},format=gray",
                        f"{TARGETS}/frame_%05d.png"], check=True, capture_output=True)
    print(len(os.listdir(TARGETS)), "frames")


def load_targets(device):
    files = sorted(f for f in os.listdir(TARGETS) if f.endswith(".png"))
    import PIL.Image
    frames = np.stack([np.array(PIL.Image.open(f"{TARGETS}/{f}"), dtype=np.float32) / 255.0 for f in files])
    p = frames + 1e-8
    del frames
    p /= p.sum(-1, keepdims=True)  # softmax targets need unit row mass
    logt = np.log(p)
    logt -= logt.mean(-1, keepdims=True)  # softmax is shift-invariant; centering removes the free offset
    return torch.from_numpy(logt).to(device)


def logits(x, wq, wk):
    q = x @ wq
    k = x @ wk
    s = q @ k.mT / math.sqrt(D)
    return s - s.mean(-1, keepdim=True)


def display_frames(attn):
    # per-row z-score then percentile clip exposes relative structure; blur kills row streaks
    z = (attn - attn.mean(-1, keepdims=True)) / (attn.std(-1, keepdims=True) + 1e-8)
    lo = np.percentile(z, 1, axis=(-2, -1), keepdims=True)
    hi = np.percentile(z, 99, axis=(-2, -1), keepdims=True)
    img = np.clip((z - lo) / (hi - lo + 1e-8), 0, 1)
    for i in range(img.shape[0]):
        img[i] = gaussian_filter(img[i], 1.5)
    return (CMAP(img)[..., :3] * 255).astype(np.uint8)


def encode_video(x, wq, wk, path, batch=128):
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{N}x{N}", "-r", str(FPS),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            s = logits(x[i:i + batch], wq, wk)
            attn = s.softmax(-1).float().cpu().numpy()
            for frame in display_frames(attn):
                proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()


def attach_audio(silent):
    out = silent.replace(".mp4", "_audio.mp4")
    subprocess.run(["ffmpeg", "-y", "-i", silent, "-i", AUDIO, "-c:v", "copy", "-c:a", "aac",
                    "-shortest", out], check=True, capture_output=True)
    os.remove(silent)
    return out


def train(args):
    torch.manual_seed(0)
    device = torch.device("mps")
    logt = load_targets(device)
    n_frames = logt.shape[0]
    print(f"{n_frames} frames on {device}")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        x = torch.nn.Parameter(ckpt["x"])
        wq = torch.nn.Parameter(ckpt["wq"])
        wk = torch.nn.Parameter(ckpt["wk"])
    else:
        x = torch.nn.Parameter(torch.randn(n_frames, N, D, device=device) * args.init_scale)
        wq = torch.nn.Parameter(torch.empty(D, D, device=device).normal_() * 0.02)
        wk = torch.nn.Parameter(torch.empty(D, D, device=device).normal_() * 0.02)
    # shared Q/K moves the target under every frame; keep them slow so frames converge
    opt = torch.optim.Adam([
        {"params": [x], "lr": args.lr},
        {"params": [wq, wk], "lr": args.lr / 10},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-4)

    os.makedirs(OUT, exist_ok=True)
    snapshot_steps = sorted({max(1, args.steps * i // args.snapshots) for i in range(1, args.snapshots + 1)})
    losses, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n_frames, (args.batch,), device=device)
        loss = F.mse_loss(logits(x[idx], wq, wk), logt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step in snapshot_steps:
            name = f"{OUT}/{args.prefix}_{step * 100 // args.steps:03d}pct.mp4"
            encode_video(x.detach(), wq.detach(), wk.detach(), name)
            attach_audio(name)
            print(f"step {step}/{args.steps} loss {np.mean(losses[-100:]):.3f} "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)

    torch.save({"x": x.data, "wq": wq.data, "wk": wk.data}, f"{OUT}/{args.save}")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 3))
    plt.plot(losses, linewidth=0.5)
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("logit MSE")
    plt.tight_layout()
    plt.savefig(f"{OUT}/loss.png", dpi=120)
    print(f"done in {time.time() - t0:.0f}s; final loss {np.mean(losses[-100:]):.3f}")


def render(args):
    device = torch.device("mps")
    logt = load_targets(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    encode_video(ckpt["x"], ckpt["wq"], ckpt["wk"], f"{OUT}/final_silent.mp4")
    print(attach_audio(f"{OUT}/final_silent.mp4"))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download").set_defaults(fn=download)
    p_train = sub.add_parser("train")
    p_train.add_argument("--steps", type=int, default=20000)
    p_train.add_argument("--batch", type=int, default=256)
    p_train.add_argument("--lr", type=float, default=0.1)
    p_train.add_argument("--snapshots", type=int, default=10)
    p_train.add_argument("--init-scale", type=float, default=0.1)
    p_train.add_argument("--resume")
    p_train.add_argument("--prefix", default="progress")
    p_train.add_argument("--save", default="final.pt")
    p_train.set_defaults(fn=train)
    p_render = sub.add_parser("render")
    p_render.add_argument("--ckpt", default=f"{OUT}/final.pt")
    p_render.set_defaults(fn=render)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
