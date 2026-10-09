"""Step 4: supervised fine-tuning. Teaches the base model to follow the chat format.
Loss is only computed on the assistant's tokens."""
import argparse
import math
import random
from contextlib import nullcontext

import torch

from common import END, enc, get_device, prompt_ids
from model import configure_optimizer, load_ckpt, save_ckpt

p = argparse.ArgumentParser()
p.add_argument("--base", default="out/base.pt")
p.add_argument("--out", default="out/chat.pt")
p.add_argument("--epochs", type=float, default=2)
p.add_argument("--batch_size", type=int, default=8)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--eval_every", type=int, default=200)
args = p.parse_args()

device = get_device()
# bf16 mixed precision on NVIDIA GPUs; plain fp32 on Mac/CPU
autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()
model, _ = load_ckpt(args.base, device)
block = model.cfg.block_size
data = torch.load("data/sft.pt")
train, val = data["train"], data["val"]
opt = configure_optimizer(model, args.lr, weight_decay=0.0)
steps = int(args.epochs * len(train) / args.batch_size)
print(f"device={device}  {len(train)} conversations  {steps} steps")


def collate(batch):
    T = min(block, max(len(ids) for ids, _ in batch) - 1)
    x = torch.full((len(batch), T), END, dtype=torch.long)
    y = torch.full((len(batch), T), -1, dtype=torch.long)
    for i, (ids, mask) in enumerate(batch):
        ids, mask = ids[:T + 1], mask[:T + 1]
        n = len(ids) - 1
        x[i, :n] = torch.tensor(ids[:-1])
        tgt = torch.tensor(ids[1:])
        tgt[torch.tensor(mask[1:]) == 0] = -1
        y[i, :n] = tgt
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate():
    model.eval()
    losses = [model(*collate(val[i:i + args.batch_size]))[1].item()
              for i in range(0, min(len(val), 20 * args.batch_size), args.batch_size)]
    q = [{"role": "user", "content": "Give me three tips for staying focused while studying."}]
    reply = enc.decode(list(model.generate(prompt_ids(q), max_new_tokens=80, stop_token=END)))
    model.train()
    return sum(losses) / len(losses), reply


model.train()
order = []
for step in range(steps):
    if len(order) < args.batch_size:
        order += random.sample(range(len(train)), len(train))
    batch = [train[order.pop()] for _ in range(args.batch_size)]
    lr = args.lr * 0.5 * (1 + math.cos(math.pi * step / steps))
    for g in opt.param_groups:
        g["lr"] = lr
    with autocast:
        _, loss = model(*collate(batch))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    if step % 20 == 0:
        print(f"step {step:5d}/{steps} | loss {loss.item():.3f} | lr {lr:.2e}")
    if step % args.eval_every == 0 or step == steps - 1:
        vl, reply = evaluate()
        print(f"--- val loss {vl:.3f}\n    Q: tips for staying focused?\n    A: {reply!r}")
        save_ckpt(args.out, model)

print(f"done, saved {args.out}")
