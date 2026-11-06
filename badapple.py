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

VIDEO_URL = "https://www.youtube.com/watch?v=FtutLA63Cp8"
FRAMES = "frames"
VIDEO = "frames/bad_apple.mp4"
AUDIO = "frames/bad_apple_audio.mp3"
TARGETS = "frames/targets"
OUT = "out"
N = 256  # token count == attention matrix side == frame resolution
D = 64   # head dim; logits stay rank-D, the same effective rank that worked for GPT-2 XL
FPS = 15
CMAP = matplotlib.colormaps["magma"]

SR = 11025
N_FFT = 1024
HOP = 256
BINS = 256          # matrix columns; higher bins (5.5k-11k Hz) are dropped
SONG_PATCH = 256    # time frames per matrix: 256 * HOP / SR ≈ 5.9s of song


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


def load_targets():
    files = sorted(f for f in os.listdir(TARGETS) if f.endswith(".png"))
    import PIL.Image
    frames = np.stack([np.array(PIL.Image.open(f"{TARGETS}/{f}"), dtype=np.float32) / 255.0 for f in files])
    p = frames + 1e-8
    del frames
    p /= p.sum(-1, keepdims=True)  # softmax targets need unit row mass
    logt = np.log(p)
    logt -= logt.mean(-1, keepdims=True)  # softmax is shift-invariant; centering removes the free offset
    return torch.from_numpy(logt)  # stays on CPU: 864MB of wired MPS memory invites jetsam


def logits(x, wq, wk):
    d = wq.shape[0]  # head dim follows the checkpoint; video and song heads differ
    q = x @ wq
    k = x @ wk
    s = q @ k.mT / math.sqrt(d)
    return s - s.mean(-1, keepdim=True)


def display_frames(attn):
    # attn: (B, N, N) torch tensor; per-row z-score + percentile clip expose relative
    # structure, blur kills row streaks. All stages run batched on the attention's device.
    z = (attn - attn.mean(-1, keepdim=True)) / (attn.std(-1, keepdim=True) + 1e-8)
    flat = z.reshape(z.shape[0], -1)
    srt = flat.sort(dim=1).values
    n1 = flat.shape[1] - 1
    p1, p2 = 0.01 * n1, 0.99 * n1  # np.percentile-compatible linear interpolation
    lo = (1 - p1 % 1) * srt[:, int(p1):int(p1) + 1] + (p1 % 1) * srt[:, int(p1) + 1:int(p1) + 2]
    hi = (1 - p2 % 1) * srt[:, int(p2):int(p2) + 1] + (p2 % 1) * srt[:, int(p2) + 1:int(p2) + 2]
    img = ((flat - lo) / (hi - lo + 1e-8)).clamp(0, 1).reshape(attn.shape)
    r = 6  # matches scipy truncate=4 default
    xs = torch.arange(-r, r + 1, dtype=torch.float32, device=attn.device)
    kern = (torch.exp(-xs**2 / 4.5) / torch.exp(-xs**2 / 4.5).sum()).view(1, 1, 1, -1)
    im = F.conv2d(F.pad(img[:, None], (r, r, 0, 0), mode="replicate"), kern)
    im = F.conv2d(F.pad(im, (0, 0, r, r), mode="replicate"), kern.transpose(-1, -2))
    q = (im[:, 0] * 255).round().long().clamp(0, 255)
    lut = torch.from_numpy((CMAP(np.linspace(0, 1, 256))[..., :3] * 255).astype(np.uint8)).to(attn.device)
    return lut[q].cpu().numpy()


def param_frames(xb):
    # raw weights as an image: no decode, just min-max per frame; shows the aux-pinned
    # half stretched back to full width so it lands in the N-wide video pipe
    img = xb[..., :N // 4]
    lo = img.amin(dim=(-2, -1), keepdim=True)
    img = ((img - lo) / (img.amax(dim=(-2, -1), keepdim=True) - lo + 1e-8)).clamp(0, 1)
    return img.repeat_interleave(N // img.shape[-1], dim=-1)


def encode_video(x, wq, wk, path, batch=128, aux=None):
    wide = aux is not None
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{(2 if wide else 1) * N}x{N}", "-r", str(FPS),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            s = logits(x[i:i + batch], wq, wk)
            frames = display_frames(s.softmax(-1))
            if wide:
                left = param_frames(aux[i:i + batch]).cpu().numpy()
                for l, r in zip(left, frames):
                    frame = np.concatenate([(l[..., None] * 255).astype(np.uint8).repeat(3, -1), r], axis=1)
                    proc.stdin.write(frame.tobytes())
            else:
                for frame in frames:
                    proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()


def frame_losses(x, wq, wk, logt):
    out = []
    with torch.no_grad():
        for i in range(0, x.shape[0], 512):
            s = logits(x[i:i + 512], wq, wk)
            t = logt[i:i + 512].to(x.device)
            out.append(((s - t) ** 2).mean((-2, -1)).cpu())
    return torch.cat(out).numpy()


def attach_audio(silent):
    out = silent.replace(".mp4", "_audio.mp4")
    subprocess.run(["ffmpeg", "-y", "-i", silent, "-i", AUDIO, "-c:v", "copy", "-c:a", "aac",
                    "-shortest", out], check=True, capture_output=True)
    os.remove(silent)
    return out


def load_aux():
    import PIL.Image
    files = sorted(f for f in os.listdir(TARGETS) if f.endswith(".png"))
    a = np.stack([np.array(PIL.Image.open(f"{TARGETS}/{f}"), dtype=np.float32) / 255.0 for f in files])
    aux = a.reshape(len(a), N, N // 4, 4).mean(-1)
    del a
    return torch.from_numpy(aux)  # CPU, same reason as load_targets


def train(args):
    torch.manual_seed(0)
    device = torch.device("mps")
    logt = load_targets()
    aux = load_aux()
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
    heat, main_losses = [], []
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n_frames, (args.batch,))
        idx_g = idx.to(device)
        main = F.mse_loss(logits(x[idx_g], wq, wk), logt[idx].to(device))
        loss = main
        if args.aux > 0:
            # first 64 dims are pinned to a squeezed view of the frame so the raw weights
            # play the video; the rest stay free for the attention basis
            loss = loss + args.aux * F.mse_loss(x[idx_g][..., :aux.shape[-1]], aux[idx].to(device))
        main_losses.append(main.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step in snapshot_steps:
            torch.save({"x": x.data, "wq": wq.data, "wk": wk.data}, f"{OUT}/{args.save}")
            torch.mps.empty_cache()
            name = f"{OUT}/{args.prefix}_{step * 100 // args.steps:03d}pct.mp4"
            encode_video(x.detach(), wq.detach(), wk.detach(), name, aux=aux)
            attach_audio(name)
            heat.append(frame_losses(x.detach(), wq.detach(), wk.detach(), logt))
            print(f"step {step}/{args.steps} main {np.mean(main_losses[-100:]):.3f} "
                  f"total {np.mean(losses[-100:]):.3f} ({time.time() - t0:.0f}s elapsed)", flush=True)

    torch.save({"x": x.data, "wq": wq.data, "wk": wk.data}, f"{OUT}/{args.save}")
    import matplotlib.pyplot as plt
    heat = np.stack(heat)
    plt.figure(figsize=(12, 3))
    plt.imshow(heat, aspect="auto", cmap="magma")
    plt.colorbar(label="per-frame logit MSE")
    plt.xlabel("frame")
    plt.ylabel("snapshot")
    plt.tight_layout()
    plt.savefig(f"{OUT}/loss_heat.png", dpi=120)
    plt.figure(figsize=(8, 3))
    plt.plot(losses, linewidth=0.5)
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("logit MSE")
    plt.tight_layout()
    plt.savefig(f"{OUT}/loss.png", dpi=120)
    print(f"done in {time.time() - t0:.0f}s; final loss {np.mean(losses[-100:]):.3f}")


def encode_params(x, path, batch=128):
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{N}x{N}", "-r", str(FPS),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            for f in param_frames(x[i:i + batch]).cpu().numpy():
                proc.stdin.write((f[..., None] * 255).astype(np.uint8).repeat(3, -1).tobytes())
    proc.stdin.close()
    proc.wait()


def render(args):
    device = torch.device("mps")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    if args.params:
        encode_params(ckpt["x"], f"{OUT}/final_params.mp4")
        print(f"{OUT}/final_params.mp4")
        return
    encode_video(ckpt["x"], ckpt["wq"], ckpt["wk"], f"{OUT}/final_silent.mp4")
    print(attach_audio(f"{OUT}/final_silent.mp4"))


def song_train(args):
    from scipy.io import wavfile
    wav = f"{FRAMES}/song.wav"
    if not os.path.exists(wav):
        subprocess.run(["ffmpeg", "-y", "-i", AUDIO, "-ar", str(SR), "-ac", "1", wav],
                       check=True, capture_output=True)
    sr, y = wavfile.read(wav)
    y = torch.from_numpy(y.astype(np.float32) / 32768.0)
    device = torch.device("mps")
    y = y.to(device)
    win = torch.hann_window(N_FFT, device=device)
    S = torch.stft(y, N_FFT, HOP, window=win, return_complex=True)
    M = S.abs().T[:, :BINS]
    mmax = M.max()
    # sqrt compression keeps quiet bins visible after log; g in [0,1] by construction
    g = (M / mmax).sqrt()
    E = g.sum(-1)  # per-time loudness; softmax rows can't carry it, so it goes through a linear readout
    p = g / (g.sum(-1, keepdim=True) + 1e-8)
    T0 = (g.shape[0] // SONG_PATCH) * SONG_PATCH
    # clamp the silence floor: matching deep -18 tails wastes capacity and adds GL noise
    logt = torch.log(p[:T0] + 1e-8).clamp(min=-8)
    logt -= logt.mean(-1, keepdim=True)
    logt = logt.reshape(-1, SONG_PATCH, BINS)
    e = torch.log(E[:T0] + 1e-6)
    mu, sd = e.mean(), e.std() + 1e-6
    ez = ((e - mu) / sd).reshape(-1, SONG_PATCH)
    P = logt.shape[0]

    torch.manual_seed(0)
    dim = args.dim
    x = torch.nn.Parameter(torch.randn(P, N, dim, device=device) * args.init_scale)
    wq = torch.nn.Parameter(torch.empty(dim, dim, device=device).normal_() * 0.02)
    wk = torch.nn.Parameter(torch.empty(dim, dim, device=device).normal_() * 0.02)
    r = torch.nn.Parameter(torch.randn(dim, device=device) * 0.01)
    opt = torch.optim.Adam([x, wq, wk, r], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-4)
    for step in range(1, args.steps + 1):
        loss = F.mse_loss(logits(x, wq, wk), logt) + 0.1 * F.mse_loss(x @ r, ez)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(f"step {step}/{args.steps} loss {loss.item():.3f}", flush=True)
    torch.save({"x": x.data, "wq": wq.data, "wk": wk.data, "r": r.data,
                "mu": mu, "sd": sd, "mmax": mmax, "length": y.shape[0]},
               f"{OUT}/{args.save}")
    print(f"{P} patches, final loss {loss.item():.3f}")

    with torch.no_grad():
        p_hat = logits(x, wq, wk).softmax(-1)
        e_hat = ((x @ r) * sd + mu).exp()
        g_hat = (p_hat * e_hat.unsqueeze(-1)).clamp(0, 1)
        mag = (g_hat * mmax.sqrt()) ** 2
    # stft of a length-L signal yields 1 + L//HOP frames; pad the tail so GL shapes stay consistent
    full = torch.zeros(1 + y.shape[0] // HOP, N_FFT // 2 + 1)
    full[:mag.numel() // BINS, :BINS] = mag.reshape(-1, BINS).cpu()
    y_hat = griffin_lim(full, y.shape[0], args.n_iters)
    y_hat = (y_hat / y_hat.abs().max() * 0.95 * 32767).short().numpy()
    wavfile.write(f"{OUT}/song.wav", SR, y_hat)
    if os.path.exists(f"{OUT}/final.mp4"):
        subprocess.run(["ffmpeg", "-y", "-i", f"{OUT}/final.mp4", "-i", f"{OUT}/song.wav",
                        "-c:v", "copy", "-c:a", "aac", "-shortest", f"{OUT}/final_attn.mp4"],
                       check=True, capture_output=True)
        print(f"{OUT}/final_attn.mp4")
    else:
        print(f"{OUT}/song.wav")


def griffin_lim(mag, length, n_iters):
    # phase is estimated by consistency iterations; the original audio is never read back
    mag = mag.cpu()
    win = torch.hann_window(N_FFT)
    phase = torch.exp(2j * math.pi * torch.rand(mag.shape[1], mag.shape[0]))
    y = torch.istft((mag.T * phase), N_FFT, HOP, window=win, length=length)
    for _ in range(n_iters):
        s = torch.stft(y, N_FFT, HOP, window=win, return_complex=True)
        y = torch.istft(mag.T * (s / (s.abs() + 1e-10)), N_FFT, HOP, window=win, length=length)
    return y


def polish(args):
    # warm-restarting shared Q/K perturbs every frame at once; instead re-roll only
    # the frames that failed, with Q/K and the good frames frozen
    import PIL.Image
    device = torch.device("mps")
    logt = load_targets()
    ckpt = torch.load(f"{OUT}/final.pt", map_location=device, weights_only=True)
    x, wq, wk = ckpt["x"], ckpt["wq"], ckpt["wk"]
    with torch.no_grad():
        corrs = []
        for i in range(0, x.shape[0], 512):
            a = logits(x[i:i + 512], wq, wk).softmax(-1)
            a -= a.mean(-1, keepdim=True)
            t = logt[i:i + 512].to(device)
            t -= t.mean(-1, keepdim=True)
            corrs.append(((a * t).sum((-2, -1)) / (a.norm(dim=(-2, -1)) * t.norm(dim=(-2, -1)) + 1e-9)).cpu())
        corr = torch.cat(corrs).numpy()
    files = sorted(f for f in os.listdir(TARGETS) if f.endswith(".png"))
    contrast = np.array([np.array(PIL.Image.open(f"{TARGETS}/{f}"), dtype=np.float32).std() for f in files])
    idx = np.where((corr < args.corr) & (contrast > args.contrast))[0]
    idx_t = torch.from_numpy(idx).to(device)
    n = len(idx)
    print(f"polishing {n} frames (corr mean {corr.mean():.3f} before)", flush=True)

    logt_bad = logt[idx_t.cpu()].to(device)
    del logt
    torch.mps.empty_cache()  # jetsam kills the process otherwise; the freed 864MB is not returned automatically
    torch.manual_seed(1)
    xb = torch.nn.Parameter(x[idx_t].clone())
    with torch.no_grad():
        xb.copy_(torch.randn_like(xb) * args.init_scale)  # fresh roll for frames stuck in bad basins
    opt = torch.optim.Adam([xb], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-4)
    for step in range(1, args.steps + 1):
        pos = torch.randint(0, n, (args.batch,), device=device)
        loss = F.mse_loss(logits(xb[pos], wq, wk), logt_bad[pos])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps:
            print(f"step {step}/{args.steps} loss {loss.item():.3f}", flush=True)
        if step % 2000 == 0:
            x[idx_t] = xb.data
            torch.save({"x": x, "wq": wq, "wk": wk}, f"{OUT}/{args.save}")
    with torch.no_grad():
        x[idx_t] = xb.data
    torch.save({"x": x, "wq": wq, "wk": wk}, f"{OUT}/{args.save}")
    print(f"saved {OUT}/{args.save}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download").set_defaults(fn=download)
    p_train = sub.add_parser("train")
    p_train.add_argument("--steps", type=int, default=20000)
    p_train.add_argument("--dim", type=int, default=128)
    p_train.add_argument("--batch", type=int, default=256)
    p_train.add_argument("--lr", type=float, default=0.1)
    p_train.add_argument("--snapshots", type=int, default=10)
    p_train.add_argument("--init-scale", type=float, default=0.1)
    p_train.add_argument("--resume")
    p_train.add_argument("--aux", type=float, default=0.2)
    p_train.add_argument("--prefix", default="progress")
    p_train.add_argument("--save", default="final.pt")
    p_train.set_defaults(fn=train)
    p_render = sub.add_parser("render")
    p_render.add_argument("--ckpt", default=f"{OUT}/final.pt")
    p_render.add_argument("--params", action="store_true")
    p_render.set_defaults(fn=render)
    p_song = sub.add_parser("song")
    p_song.add_argument("--steps", type=int, default=6000)
    p_song.add_argument("--dim", type=int, default=256)
    p_song.add_argument("--lr", type=float, default=0.1)
    p_song.add_argument("--init-scale", type=float, default=0.1)
    p_song.add_argument("--save", default="song.pt")
    p_song.add_argument("--n-iters", type=int, default=100)
    p_song.set_defaults(fn=song_train)
    p_polish = sub.add_parser("polish")
    p_polish.add_argument("--corr", type=float, default=0.5)
    p_polish.add_argument("--contrast", type=float, default=60.0)
    p_polish.add_argument("--steps", type=int, default=5000)
    p_polish.add_argument("--batch", type=int, default=256)
    p_polish.add_argument("--lr", type=float, default=0.1)
    p_polish.add_argument("--init-scale", type=float, default=0.1)
    p_polish.add_argument("--save", default="polished.pt")
    p_polish.set_defaults(fn=polish)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
