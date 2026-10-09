"""Shared helpers: tokenizer, chat format, device selection."""
import torch
import tiktoken

# GPT-2 BPE plus three chat tokens.
_base = tiktoken.get_encoding("gpt2")
SPECIAL = {"<|endoftext|>": 50256, "<|user|>": 50257, "<|assistant|>": 50258, "<|end|>": 50259}
enc = tiktoken.Encoding(
    name="gpt2_chat",
    pat_str=_base._pat_str,
    mergeable_ranks=_base._mergeable_ranks,
    special_tokens=SPECIAL,
)
EOT, USER, ASSISTANT, END = (SPECIAL[k] for k in ("<|endoftext|>", "<|user|>", "<|assistant|>", "<|end|>"))
VOCAB_SIZE = 50304  # 50260 padded up to a multiple of 64 for speed


def encode_chat(messages):
    """messages: [{"role": "user"|"assistant", "content": str}, ...]
    Returns (ids, mask) where mask=1 marks tokens the model should learn to produce."""
    ids, mask = [], []
    for m in messages:
        role_tok = USER if m["role"] == "user" else ASSISTANT
        body = enc.encode_ordinary(m["content"].strip()) + [END]
        ids += [role_tok] + body
        learn = 1 if m["role"] == "assistant" else 0
        mask += [0] + [learn] * len(body)
    return ids, mask


def prompt_ids(messages):
    """Tokens for a conversation so far, ending with the assistant tag so the model replies."""
    ids, _ = encode_chat(messages)
    return ids + [ASSISTANT]


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
