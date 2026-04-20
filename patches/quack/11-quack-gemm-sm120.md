# Patch 11: quack SM120 GEMM

**Priority: Medium** — Custom BF16/FP16 dense GEMM for SM120.

## Fix

`quack/gemm_sm120.py` — A `GemmSm120(GemmSm90)` subclass implementing SM120-compatible GEMM using:
- `warp.MmaF16BF16Op` (32 threads) instead of WGMMA (128 threads)
- `ldmatrix` for SMEM→register copies
- 99 KB SMEM budget
- Optional pingpong scheduling

Based on CUTLASS example `blackwell_geforce/dense_gemm.py`.

**Limitation**: BF16/FP16 only. FP8 not supported via warp-level MMA on SM120.
