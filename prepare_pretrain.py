"""Step 1: stream public web text (FineWeb-Edu) and tokenize it into data/train.bin + data/val.bin."""
import argparse
import os
import sys

import numpy as np
from datasets import load_dataset

from common import EOT, enc

p = argparse.ArgumentParser()
p.add_argument("--tokens", type=float, default=100e6, help="how many tokens to collect (default 100M)")
p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
p.add_argument("--subset", default="sample-10BT")
args = p.parse_args()

os.makedirs("data", exist_ok=True)
target = int(args.tokens)
val_target = max(target // 100, 100_000)

ds = load_dataset(args.dataset, name=args.subset, split="train", streaming=True)
buf = np.empty(target + val_target + 1_000_000, dtype=np.uint16)
n, batch = 0, []


def flush():
    global n, batch
    for toks in enc.encode_ordinary_batch(batch):
        toks.append(EOT)
        k = min(len(toks), len(buf) - n)
        buf[n:n + k] = toks[:k]
        n += k
    batch = []


for ex in ds:
    batch.append(ex["text"])
    if len(batch) == 512:
        flush()
        print(f"\r{n / 1e6:.1f}M / {(target + val_target) / 1e6:.0f}M tokens", end="", flush=True)
        if n >= target + val_target:
            break
if batch:
    flush()
print()

val, train = buf[:val_target], buf[val_target:n]
train.tofile("data/train.bin")
val.tofile("data/val.bin")
print(f"wrote data/train.bin ({len(train) / 1e6:.1f}M tokens) and data/val.bin ({len(val) / 1e6:.2f}M tokens)")

# the streaming dataset can leave background threads that block a normal exit
sys.stdout.flush()
os._exit(0)
