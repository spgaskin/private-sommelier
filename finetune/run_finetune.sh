#!/usr/bin/env bash
# QLoRA fine-tune on Apple Silicon via MLX-LM. Tunes BEHAVIOR (house style:
# cite sources, refuse to guess), while FACTS stay in the RAG layer (../rag.py).
#
# mlx-lm is pulled on demand with `uv run --with` so it never bloats the main
# project deps. The base model is downloaded from Hugging Face on first run.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${FT_MODEL:-mlx-community/Qwen2.5-3B-Instruct-4bit}"
ITERS="${FT_ITERS:-200}"

echo ">> fine-tuning $MODEL for $ITERS iters (adapters -> ./adapters)"
uv run --with mlx-lm python -m mlx_lm lora \
  --model "$MODEL" \
  --train \
  --data ./data \
  --iters "$ITERS" \
  --batch-size 1 \
  --num-layers 8 \
  --adapter-path ./adapters

echo
echo ">> done. try the tuned behavior:"
echo "uv run --with mlx-lm python -m mlx_lm generate --model $MODEL \\"
echo "    --adapter-path ./adapters --prompt 'Estimate our churn rate.'"
