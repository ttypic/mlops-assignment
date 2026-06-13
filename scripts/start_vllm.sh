#!/usr/bin/env bash
#
# Start vLLM with your chosen configuration.
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

set -euo pipefail

# Public model id the agent talks to (stays canonical via --served-model-name
# even though we load the FP8 checkpoint). Matches VLLM_MODEL in .env.
SERVED_NAME="Qwen/Qwen3-30B-A3B-Instruct-2507"

# Official FP8 checkpoint: ~half the weight memory of BF16 (~31GB vs ~61GB),
# which is the single biggest lever here - it buys ~30GB of extra KV cache on
# the 80GB H100, and that headroom is what lets us hit 10 RPS. FP8 on Hopper is
# near-lossless at this model size. Swap back to the BF16 id to A/B quality.
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 256 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192