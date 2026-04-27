#!/bin/bash
# =============================================================================
# Kimi-K2.6 INT4 inference on RTX PRO 6000 × 6 (SM120)
# =============================================================================
set -euo pipefail

IMAGE="lna-lab/kimi-k26-inference:latest"
CONTAINER="kimi-k26"
MODEL_PATH="/media/tonoken/SN8100/Kimi-K2.6-NVFP4A16-LnaLab-routed"
PORT=8016

# Build if image doesn't exist
if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "Building $IMAGE..."
    docker build \
        -f /media/tonoken/P4800X/Lna-Lab/blackwell-geforce-nvfp4-gemm/Dockerfile.kimi-k26 \
        -t "$IMAGE" \
        /media/tonoken/P4800X/Lna-Lab/blackwell-geforce-nvfp4-gemm
fi

# Stop existing container
docker rm -f "$CONTAINER" 2>/dev/null || true

echo "Starting Kimi-K2.6 on GPU 0-5, port $PORT..."
docker run -d \
    --name "$CONTAINER" \
    --runtime nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -v "$MODEL_PATH":/models/current:ro \
    -p "$PORT":8016 \
    --shm-size 64g \
    "$IMAGE" \
    --trust-remote-code

echo ""
echo "Container: $CONTAINER"
echo "Logs: docker logs -f $CONTAINER"
echo "Test: curl http://localhost:$PORT/v1/models"
