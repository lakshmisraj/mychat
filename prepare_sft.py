"""Step 3: download public chat conversations (SmolTalk) and tokenize them with the chat template."""
import argparse
import os
import sys
import random

import torch
from datasets import load_dataset

from common import encode_chat

p = argparse.ArgumentParser()
p.add_argument("--examples", type=int, default=50_000)
p.add_argument("--max_len", type=int, default=512, help="drop conversations longer than this many tokens")
p.add_argument("--name", default="MiniChat", help="what your assistant calls itself")
p.add_argument("--dataset", default="HuggingFaceTB/smol-smoltalk")
args = p.parse_args()

# A few hand-written conversations so the model knows who it is.
IDENTITY = [
    ("Who are you?", f"I'm {args.name}, a small language model trained from scratch. How can I help?"),
    ("What is your name?", f"My name is {args.name}."),
    ("Are you ChatGPT?", f"No, I'm {args.name}, a small model trained from scratch on public data."),
    ("hi", f"Hi! I'm {args.name}. What can I do for you today?"),
    ("Hello!", "Hello! How can I help you today?"),
    ("What can you do?", f"I'm {args.name}. I can chat, answer simple questions, and help with writing. "
                         "I'm small, so double-check anything important."),
]


def clean(messages):
    out, system = [], ""
    for m in messages:
        if m["role"] == "system":
            system = m["content"].strip()
        elif m["role"] in ("user", "assistant") and m["content"].strip():
            out.append({"role": m["role"], "content": m["content"]})
    if system and out and out[0]["role"] == "user":
        out[0] = {"role": "user", "content": system + "\n\n" + out[0]["content"]}
    # must alternate user/assistant, starting with user
    if not out or any(m["role"] != ("user" if i % 2 == 0 else "assistant") for i, m in enumerate(out)):
        return None
    return out


examples = []
ds = load_dataset(args.dataset, split="train", streaming=True)
for ex in ds:
    msgs = clean(ex["messages"])
    if msgs is None:
        continue
    ids, mask = encode_chat(msgs)
    if len(ids) <= args.max_len:
        examples.append((ids, mask))
    if len(examples) % 2000 == 0:
        print(f"\r{len(examples)} conversations", end="", flush=True)
    if len(examples) >= args.examples:
        break
print()

for q, a in IDENTITY:
    for _ in range(max(1, args.examples // 2000)):  # upweight identity a bit
        examples.append(encode_chat([{"role": "user", "content": q}, {"role": "assistant", "content": a}]))

random.seed(0)
random.shuffle(examples)
n_val = max(200, len(examples) // 50)
os.makedirs("data", exist_ok=True)
torch.save({"train": examples[n_val:], "val": examples[:n_val]}, "data/sft.pt")
print(f"wrote data/sft.pt: {len(examples) - n_val} train / {n_val} val conversations")

# the streaming dataset can leave background threads that block a normal exit
sys.stdout.flush()
os._exit(0)
