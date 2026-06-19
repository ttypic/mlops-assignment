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

# Flag rationale (see REPORT.md for the longer version):
#   --tensor-parallel-size 1   single H100; the 3.3B active MoE path fits on one
#                              GPU, so TP would only add comms overhead.
#   --max-model-len 8192       our prompts top out ~3K + a few-hundred-token
#                              output; the model's native 262K context would
#                              force a tiny KV cache. Capping at 8K shrinks the
#                              per-sequence KV footprint and raises concurrency.
#   --gpu-memory-utilization 0.92  use the H100 fully for KV cache; leaves a
#                              safety margin for activations / fragmentation.
#   --max-num-seqs 256         short context + FP8 leave plenty of KV headroom,
#                              so allow a deep running batch to reach 10 RPS.
#   --enable-prefix-caching    the system prompt + per-DB schema prefix is reused
#                              across the 2-3 calls in a request AND across
#                              questions hitting the same DB -> big TTFT win.
#   --enable-chunked-prefill   interleave the 1.5-3K-token prefills with ongoing
#                              decodes so prefill bursts don't stall TTFT/ITL
#                              under load.
#   --max-num-batched-tokens 8192  token budget per step; balances prefill
#                              chunk size against decode latency. A Phase 6 lever.
#
# Levers intentionally left for Phase 6 (change one thing, measure, see REPORT):
#   --kv-cache-dtype fp8       more KV headroom at a small quality risk.
#   --max-num-batched-tokens   raise for throughput / lower for tighter ITL.
#   --max-num-seqs             cap to bound queue depth if P95 latency suffers.
