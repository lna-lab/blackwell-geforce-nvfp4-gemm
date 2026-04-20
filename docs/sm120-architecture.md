# SM120 Architecture: The Blackwell GeForce Chimera

## Overview

SM120 (Blackwell GeForce / RTX PRO) is not a simplified version of SM100 (Blackwell Datacenter). It is a distinct architecture that combines features from three GPU generations into something that has no direct precedent.

This document is based on empirical testing on 7× RTX PRO 6000 Blackwell Workstation Edition GPUs, CUDA 13.0, and CUTLASS 4.0 source analysis.

## ISA Comparison

### MMA Instructions

| Architecture | MMA Instruction | Operand Source | Tile Shape (FP8) |
|-------------|----------------|----------------|-----------------|
| SM80 (Ampere) | `mma.sync.aligned.m16n8k16` | Registers | 16×8×16 |
| SM90 (Hopper) | `wgmma.mma_async` | SMEM + Registers | 64×N×16 |
| SM100 (B200) | `tcgen05.mma` (UMMA) | Tensor Memory | 128×N×32 |
| **SM120 (GeForce)** | **`mma.sync.aligned.kind::f8f6f4.m16n8k32`** | **Registers** | **16×8×32** |

SM120's MMA instruction is derived from SM80's `mma.sync` family but with extended type support (`f8f6f4` kinds — FP8, FP6, FP4). It is synchronous, warp-level (32 threads), and register-to-register.

### What SM120 Does NOT Have

These SM100 features are completely absent:

1. **Tensor Memory (TMEM)** — SM100's on-chip scratchpad for MMA operands. SM120 has no TMEM; all data goes through registers.
2. **UMMA (Unified MMA)** — SM100's descriptor-based MMA that reads from TMEM. SM120 uses register-based MMA.
3. **`tcgen05.*` instruction family** — `tcgen05.mma`, `tcgen05.fence`, `tcgen05.commit`, `tcgen05.wait` — none of these exist on SM120.
4. **Cluster multicast TMA** — SM100 can broadcast TMA loads to multiple CTAs in a cluster. SM120 clusters must be 1×1×1.

### Empirical Proof

We attempted to compile SM100 kernels for SM120:

1. **`sm_120a` target**: ptxas rejects `tcgen05.mma` and `tcgen05.fence` as "not supported on .target sm_120a"
2. **`sm_100f` target** (Blackwell family): Compilation succeeds, but `cuLibraryLoadFromFile` fails to load the cubin on SM120 hardware — the driver rejects the binary as incompatible
3. **Conclusion**: SM120 and SM100 are binary-incompatible at the ISA level

## Shared Memory

| Architecture | Shared Memory per SM | Shared Memory per Block |
|-------------|---------------------|------------------------|
| SM80 | 163 KB | 48 KB (default), 163 KB (opt-in) |
| SM90 | 228 KB | 228 KB |
| SM100 | 228 KB | 228 KB |
| **SM120** | **99 KB** | **49 KB (default), 99 KB (opt-in)** |

The 99 KB limit is the single most important constraint for kernel design on SM120. Every tile size, pipeline stage count, and buffer allocation must be designed to fit within this budget.

### Impact on GEMM Tile Sizes

For NVFP4 (W4A4) with FP8 scales and BF16 output:

```
Per tile shared memory ≈ (BLOCK_M × BLOCK_K × 0.5)  [A, FP4]
                       + (BLOCK_N × BLOCK_K × 0.5)  [B, FP4]
                       + scale factor buffers
                       + barrier/metadata overhead
```

Safe SM120 tile configurations (empirically validated):
- `[128, 128, 128]` — primary tile
- `[128, 128, 256]` — extended K
- `[256, 128, 128]` — extended M
- `[128, 256, 128]` — extended N

Tiles that fail on SM120 (work on SM100):
- M64 with certain N/K combinations that exceed 99 KB with multi-stage pipelines
- Any configuration requiring > 99 KB shared memory

## Data Flow

### SM100 (Datacenter Blackwell)
```
GMEM → TMA → SMEM → TMEM → UMMA (tcgen05.mma) → TMEM → SMEM → TMA → GMEM
```

### SM120 (GeForce Blackwell)
```
GMEM → TMA → SMEM → ldmatrix → Registers → MMA (mma.sync) → Registers → SMEM → TMA → GMEM
```

Key difference: SM120 requires an explicit `ldmatrix` step to move data from shared memory to registers before every MMA instruction. SM100 bypasses this entirely via Tensor Memory.

## Pipeline Architecture

SM120 uses `PipelineTmaAsync` (same as SM90 Hopper), not SM100's TMEM-based pipeline:

- **Producer warps**: Issue TMA loads from GMEM to SMEM
- **Consumer warps**: Wait on barriers, `ldmatrix` from SMEM to registers, execute `mma.sync`
- **Synchronization**: `NamedBarrier::sync` between pipeline stages (not `tcgen05.fence`)
- **Pipeline stages**: Typically 2-4, constrained by 99 KB SMEM budget

## Block Scaling

SM120 supports block-scaled MMA natively:

```
mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
```

Scale factors (UE8M0 format) are passed as register operands directly to the MMA instruction. This is the same format as SM100's block scaling but delivered through registers instead of TMEM descriptors.

## CUTLASS 4.0 SM120 Support

CUTLASS 4.0 has full SM120 support under `cutlass::arch::Sm120`:

| Component | File |
|-----------|------|
| MMA atoms | `cute/arch/mma_sm120.hpp` |
| MMA traits | `cute/atom/mma_traits_sm120.hpp` |
| Dense collective | `cutlass/gemm/collective/sm120_mma_tma.hpp` |
| Block-scaled collective | `cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp` |
| Builders | `cutlass/gemm/collective/builders/sm120_*.inl` |
| Example | `examples/87_blackwell_geforce_gemm_blockwise/` |

The CollectiveBuilder for SM120 uses:
- `OpClassTensorOp` (not `OpClassBlockScaledTensorOp`)
- Layout specified as `cute::tuple<LayoutA, LayoutSFA>` for blockwise scaling
- Cluster shape must be `1×1×1`
- Tile shapes auto-selected to fit 99 KB SMEM

## Performance Characteristics

Measured on RTX PRO 6000 (SM120, 96 GB):

### CUTLASS FP8 Dense GEMM (example 87a)

| Size | Runtime | TFLOPS |
|------|---------|--------|
| 1024×1024×1024 | 0.010 ms | 206 |
| 2048×2048×2048 | 0.033 ms | 522 |
| 4096×4096×4096 | 0.184 ms | 746 |
| 4096×4096×8192 | 0.357 ms | 771 |

### NVFP4 MoE Inference (vLLM, end-to-end)

| Model | Active Params | tok/s |
|-------|--------------|-------|
| Qwen3.6-35B MoE | 3B | 175 |
| Gemma4-26B MoE | 3.8B | 160 |

## Key Takeaway

When writing kernels for SM120:

1. **Start from SM80 MMA, not SM100 UMMA** — your MMA instruction is `mma.sync.aligned`, not `tcgen05.mma`
2. **Budget 99 KB shared memory** — use `cutlass.utils.get_smem_capacity_in_bytes("sm_120")`
3. **Use TMA for loads** — SM120 has TMA (from SM90), use it for efficient GMEM→SMEM transfers
4. **Cluster shape 1×1×1** — no multicast, each CTA loads its own data
5. **`ldmatrix` for SMEM→register** — SM75's `ldmatrix` is the path from shared memory to MMA operands
6. **Block scaling in the MMA instruction** — scale factors go in registers, not descriptors
