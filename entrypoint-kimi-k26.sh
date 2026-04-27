#!/bin/bash
# =============================================================================
# Kimi-K2.6 Inference Entrypoint — Lna-Lab
# =============================================================================
set -euo pipefail

echo "============================================"
echo "  Lna-Lab Kimi-K2.6 Inference Container"
echo "============================================"
echo "  Model:    ${MODEL_PATH}"
echo "  Port:     ${PORT}"
echo "  PP:       ${PIPELINE_PARALLEL_SIZE}"
echo "  TP:       ${TENSOR_PARALLEL_SIZE}"
echo "  MaxLen:   ${MAX_MODEL_LEN}"
echo "  Seqs:     ${MAX_NUM_SEQS}"
echo "  GPU Mem:  ${GPU_MEMORY_UTILIZATION}"
echo "  Quant:    ${QUANTIZATION}"
echo "  CUDAGraph: ${VLLM_CUDA_GRAPH_MODE:-default}"
echo "============================================"

if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model directory not found: ${MODEL_PATH}"
    echo "Mount your model directory with -v /path/to/model:/models/current"
    exit 1
fi

ARGS=(
    "${MODEL_PATH}"
    --port "${PORT}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}"
    --dtype auto
    --kv-cache-dtype auto
    --cpu-offload-gb 20
)

if [ "${QUANTIZATION}" != "auto" ]; then
    ARGS+=(--quantization "${QUANTIZATION}")
fi

MODEL_NAME=$(basename "${MODEL_PATH}")
ARGS+=(--served-model-name "${MODEL_NAME}")

# Extra args from CMD
if [ $# -gt 0 ]; then
    ARGS+=("$@")
fi

echo ""
echo "Starting vLLM: vllm serve ${ARGS[*]}"
echo ""

exec vllm serve "${ARGS[@]}"
