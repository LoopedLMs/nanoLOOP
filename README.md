
# nanoLOOP

A fork of [nanoGPT](https://github.com/karpathy/nanoGPT) that implements **looped (depth-recurrent) transformers**. The recurrent block's weights are shared across loop iterations, giving you deeper effective networks with fewer parameters. The code is plain and readable: `train.py` is a ~300-line training loop and `model.py` a ~300-line model definition.

## architecture

The model supports three configurations via `GPTConfig`:

| Config | Description |
|--------|-------------|
| **Standard** | `n_prelude=0, n_block=N, n_coda=0, n_loop=1` — equivalent to vanilla GPT |
| **Full loop** | `n_prelude=0, n_block=N, n_coda=0, n_loop=K` — all layers looped K times |
| **Prelude/Recur/Coda** | `n_prelude=P, n_block=N, n_coda=C, n_loop=K` — unique entry/exit layers with a shared recurrent core |

Key features:
- **Input injection**: Controls how the prelude output combines with loop state (`inject`, `inject_random`, `passthrough`)
- **Truncated BPTT**: `bptt_k` limits gradient flow depth through the recurrence to save memory
- **RMSNorm** between loop iterations to prevent activation blowup

## install

```
uv sync
```

## quick start

Train a character-level model on Shakespeare. First, prepare the data:

```sh
python data/shakespeare_char/prepare.py
```

**Vanilla GPT** (6 layers, no looping):

```sh
python train.py config/train_shakespeare_char.py
```

**Looped model** (2 prelude + 2 block x4 loops + 2 coda = 12 effective layers):

```sh
python train.py config/train_shakespeare_char_looped.py
```

Sample from a trained model:

```sh
python sample.py --out_dir=out-shakespeare-char-looped
```

**CPU only** (no GPU):

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_block=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

On Apple Silicon, add `--device=mps`.

## reproducing GPT-2

Tokenize [OpenWebText](https://huggingface.co/datasets/openwebtext):

```sh
python data/openwebtext/prepare.py
```

Then train (8x A100 40GB):

```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

## sampling / inference

```sh
python sample.py \
    --out_dir=out-shakespeare-char \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```

You can also prompt from a file: `python sample.py --start=FILE:prompt.txt`.

## efficiency notes

`bench.py` benchmarks the core training loop. The `estimate_mfu` function accounts for looping — forward FLOPs scale with `n_loop`, backward FLOPs scale with `min(bptt_k, n_loop)`.

## acknowledgements

Based on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy. Looped transformer architecture inspired by [Geiping et al., "Scaling up Test-Time Compute with Latent Reasoning"](https://arxiv.org/abs/2502.05171).
