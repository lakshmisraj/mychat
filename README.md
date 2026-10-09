# mychat — a ChatGPT-style model trained from scratch

Same recipe as ChatGPT, at toy scale:

| Step | Script | Data | What it learns |
|---|---|---|---|
| 1 | `prepare_pretrain.py` | [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (public educational web text) | — |
| 2 | `pretrain.py` | ↑ | Language: grammar, facts, how text flows |
| 3 | `prepare_sft.py` | [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) (public chat conversations) | — |
| 4 | `sft.py` | ↑ | To act like an assistant: answer the user, then stop |
| 5 | `chat.py` | — | You talk to it |

Model (`model.py`): Llama-style decoder-only transformer with RMSNorm, rotary position embeddings, SwiGLU, and tied embeddings.
Tokenizer: GPT-2 BPE plus `<|user|>`, `<|assistant|>`, `<|end|>` chat tokens. During fine-tuning, the loss is computed only on the assistant's replies.

## Setup
```bash
python3 -m venv .venv && .venv/bin/pip install torch numpy tiktoken datasets
```

## On this Mac (M3, 16 GB): about 4–5 hours total
Measured speed for the `tiny` preset here is about 2,600 tokens/s. Close other apps first; memory pressure slows training a lot.
```bash
.venv/bin/python prepare_pretrain.py --tokens 25e6
.venv/bin/python pretrain.py --preset tiny --max_steps 1500
.venv/bin/python prepare_sft.py --examples 20000 --max_len 256 --name MiniChat
.venv/bin/python sft.py --epochs 1
.venv/bin/python chat.py
```
Expect a model that writes fluent-looking English and follows the chat format, but often gets facts wrong. That's normal for ~16M parameters.
Training saves checkpoints as it goes. To continue an interrupted pretraining run, add `--resume`.

## On a rented NVIDIA GPU (much better results)
On CUDA the scripts automatically use bf16. On one A100/H100:
```bash
python prepare_pretrain.py --tokens 1e9
python pretrain.py --preset medium --batch_size 32 --grad_accum 4 --max_steps 8000
python prepare_sft.py --examples 200000
python sft.py --batch_size 32 --epochs 2
python chat.py
```
`medium` (~124M params, GPT-2 small size) on ~1B tokens takes a few hours and gives noticeably more coherent answers.

## Ideas to go further
- Train your own BPE tokenizer with a smaller vocabulary (e.g. 16k). It speeds up small models a lot.
- Add a KV cache to `generate()` for faster chatting.
- After SFT, add preference tuning (DPO) on a public preference dataset.
- Evaluate on a benchmark such as HellaSwag or ARC-Easy and track the score across runs.
