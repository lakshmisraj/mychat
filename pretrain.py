"""Step 2: pretrain the base model with next-token prediction on data/train.bin."""
import argparse
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch

from common import EOT, VOCAB_SIZE, enc, get_device
from model import GPT, PRESETS, GPTConfig, configure_optimizer, save_ckpt

p = argparse.ArgumentParser()
p.add_argument("--preset", default="small", choices=PRESETS)
p.add_argument("--batch_size", type=int, default=8)
p.add_argument("--grad_accum", type=int, default=8)
p.add_argument("--max_steps", type=int, default=6000)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--warmup", type=int, default=200)
p.add_argument("--eval_every", type=int, default=250)
p.add_argument("--out", default="out/base.pt")
p.add_argument("--resume", action="store_true")
args = p.parse_args()

device = get_device()
# bf16 mixed precision on NVIDIA GPUs; plain fp32 on Mac/CPU
autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
torch.manual_seed(1337)
cfg = GPTConfig(vocab_size=VOCAB_SIZE, **PRESETS[args.preset])
model = GPT(cfg).to(device)
opt = configure_optimizer(model, args.lr)
start = 0
if args.resume and os.path.exists(args.out):
    ck = torch.load(args.out, map_location=device)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optim"])
    start = ck["step"] + 1
print(f"device={device}  params={model.num_params() / 1e6:.1f}M  "
      f"tokens/step={args.batch_size * args.grad_accum * cfg.block_size:,}")

data = {s: np.memmap(f"data/{s}.bin", dtype=np.uint16, mode="r") for s in ("train", "val")}


def get_batch(split):
    d = data[split]
    ix = np.random.randint(0, len(d) - cfg.block_size - 1, args.batch_size)
    x = torch.stack([torch.from_numpy(d[i:i + cfg.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(d[i + 1:i + 1 + cfg.block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def lr_at(step):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    t = (step - args.warmup) / max(1, args.max_steps - args.warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


@torch.no_grad()
def evaluate():
    model.eval()
    loss = sum(model(*get_batch("val"))[1].item() for _ in range(20)) / 20
    sample = enc.decode(list(model.generate(enc.encode_ordinary("The most important thing about"),
                                            max_new_tokens=40, stop_token=EOT)))
    model.train()
    return loss, sample


os.makedirs(os.path.dirname(args.out), exist_ok=True)
model.train()
t0 = time.time()
for step in range(start, args.max_steps):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    for _ in range(args.grad_accum):
        with autocast:
            _, loss = model(*get_batch("train"))
        (loss / args.grad_accum).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)

    if step % 20 == 0:
        dt = time.time() - t0
        tps = 20 * args.batch_size * args.grad_accum * cfg.block_size / dt if step > start else 0
        print(f"step {step:5d} | loss {loss.item():.3f} | lr {lr_at(step):.2e} | {tps:,.0f} tok/s")
        t0 = time.time()
    if step % args.eval_every == 0 or step == args.max_steps - 1:
        vl, sample = evaluate()
        print(f"--- val loss {vl:.3f} | sample: The most important thing about{sample!r}")
        save_ckpt(args.out, model, {"optim": opt.state_dict(), "step": step})
        t0 = time.time()

print(f"done, saved {args.out}")
