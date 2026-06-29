# Fine-tuning scaffold (MLX QLoRA, Apple Silicon)

This is the report's "next phase": **tune the model for behavior, keep facts in
retrieval.** Nothing here bakes facts into weights — the facts live in `../kb/` and
are served by `../rag.py`. What we tune is *house style*: cite the source when stating
a fact, and refuse to guess numbers that aren't in the knowledge base (the exact
failure mode the build surfaced).

## What's here

- `data/train.jsonl`, `data/valid.jsonl` — behavioral examples in chat format.
- `run_finetune.sh` — one command to QLoRA-tune via MLX and write LoRA adapters.

## Prerequisites

- Apple Silicon (you have an M5 — good).
- `mlx-lm` is pulled on demand by the script (`uv run --with mlx-lm ...`), so it never
  bloats the main project. The base model downloads from Hugging Face on first run.

## Run it

```bash
./run_finetune.sh                       # ~a few minutes for 200 iters on a 3B-4bit
# pick a different base or length:
FT_MODEL=mlx-community/Qwen3-8B-4bit FT_ITERS=300 ./run_finetune.sh
```

Then compare base vs tuned behavior:

```bash
# tuned (should cite a source / refuse to guess)
uv run --with mlx-lm python -m mlx_lm generate \
    --model mlx-community/Qwen2.5-3B-Instruct-4bit \
    --adapter-path ./adapters --prompt "Estimate our churn rate."
```

## How this plugs into the agent

The fine-tune produces LoRA `adapters/`. To serve the tuned model through the same
local OpenAI endpoint, fuse the adapters and import the result into Ollama:

```bash
uv run --with mlx-lm python -m mlx_lm fuse \
    --model mlx-community/Qwen2.5-3B-Instruct-4bit \
    --adapter-path ./adapters --save-path ./fused-model
# then build an Ollama model from ./fused-model (Modelfile FROM ./fused-model)
# and point the agent at it:  PRIVATE_LLM_MODEL=acme-tuned uv run ../private_agent.py ...
```

Behavior comes from the tune; facts come from `retrieve` (RAG). That separation is the
whole point — facts stay updatable and auditable instead of stale inside the weights.
