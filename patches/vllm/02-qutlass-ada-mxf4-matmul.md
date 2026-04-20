# Patch 2: QUTLASS SM120 MXFP4 Matmul

**Priority: Critical** — Required for NVFP4 dense linear layers.

## Problem

The standard CUTLASS MXFP4 matmul uses WGMMA (warp-group MMA) instructions that are not available on SM120. Dense linear layers (QKV projections, output projections) need an SM120-native matmul path.

## Fix

`matmul_ada_mxf4_bf16_tn` — an SM120-optimized MXFP4 matrix multiply compiled into `vllm/_C.abi3.so`.

Key design:
- Uses warp-level `mma.sync.aligned` (not WGMMA)
- `ldmatrix` for SMEM→register data movement
- SMEM capacity capped at 99 KB
- Compiled with `-arch sm_120`

Called via `torch.ops._qutlass_C.matmul_ada_mxf4_bf16_tn`.

## Location

Binary: `vllm/_C.abi3.so`
Python wrapper: `vllm/_custom_ops.py` (lines 3341-3384)

## Impact

Without: NVFP4 dense layers fall back to dequantize-then-BF16-matmul → **~3× slower**
With: Native SM120 FP4 matmul → full speed
