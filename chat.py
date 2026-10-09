"""Step 5: talk to your model in the terminal.  Commands: /reset clears history, /quit exits."""
import argparse

from common import END, enc, get_device, prompt_ids
from model import load_ckpt

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="out/chat.pt")
p.add_argument("--temperature", type=float, default=0.7)
p.add_argument("--top_k", type=int, default=40)
p.add_argument("--max_tokens", type=int, default=300)
args = p.parse_args()

model, _ = load_ckpt(args.ckpt, get_device())
model.eval()
history = []
print("Chat ready. /reset to clear history, /quit to exit.\n")

while True:
    try:
        text = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if text == "/quit":
        break
    if text == "/reset":
        history = []
        print("(history cleared)\n")
        continue
    if not text:
        continue
    history.append({"role": "user", "content": text})
    # keep the most recent context, leaving room for the reply (at least half the window stays as context)
    keep = max(model.cfg.block_size - args.max_tokens, model.cfg.block_size // 2)
    ids = prompt_ids(history)[-keep:]
    print("bot> ", end="", flush=True)
    out, printed = [], 0
    for tok in model.generate(ids, args.max_tokens, args.temperature, args.top_k, stop_token=END):
        out.append(tok)
        text_so_far = enc.decode(out)
        if not text_so_far.endswith("�"):  # wait until multi-byte characters are complete
            print(text_so_far[printed:], end="", flush=True)
            printed = len(text_so_far)
    print("\n")
    history.append({"role": "assistant", "content": enc.decode(out)})
