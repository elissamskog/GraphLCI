#!/bin/bash
# setup_mlx.sh - Setup MLX backend for Mac M4
set -e

echo "--- MLX Setup Started ---"

# 1. Ensure .venv exists and is active
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# 2. Install dependencies
echo "Installing mlx-lm and huggingface_hub..."
pip install mlx-lm huggingface_hub PyYAML requests

# 3. Pull model if not exists
MODEL_ID="mlx-community/Qwen2.5-14B-Instruct-4bit"
echo "Downloading model: $MODEL_ID"

# Use huggingface-cli to download (ensures it's in the cache)
huggingface-cli download $MODEL_ID

echo "--- MLX Setup Complete ---"
echo "To use MLX, update config.yaml to set 'provider: mlx' under 'llm:'"
echo "Then run evaluation with: source .venv/bin/activate && python -m evaluation.run_evaluation_v5 ..."
