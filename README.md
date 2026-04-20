# blackwell-geforce-nvfp4-gemm

**NVFP4 inference on Blackwell GeForce / RTX PRO 6000 — patches, methodology, and benchmarks.**

RTX 5090, 5080, 5070 Ti, and RTX PRO 6000 GPUs (SM120) share a unique architecture that is *not* a subset of datacenter Blackwell (SM100). This repository documents the patches needed to run NVFP4 quantized models at full speed on these GPUs, and explains *why* each patch is necessary.

## Try it now

```bash
# Tested model (MoE, 168 tok/s on single RTX PRO 6000):
# https://huggingface.co/sakamakismile/Huihui-Qwen3.6-35B-A3B-abliterated-NVFP4

docker run -d --rm \
  --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  -v /path/to/Huihui-Qwen3.6-35B-A3B-abliterated-NVFP4:/models/current:ro \
  -p 8090:8090 \
  lna-lab/gemma4-inference:latest \
  --model /models/current --port 8090 --max-model-len 4096
```

## The SM120 Chimera

SM120 is a hybrid architecture that borrows from three generations but matches none of them:

| Feature | Source | SM120 Implementation |
|---------|--------|---------------------|
| MMA Instructions | SM80 (Ampere) | `mma.sync.aligned.m16n8k32` — warp-level, register-to-register |
| TMA (Tensor Memory Access) | SM90 (Hopper) | Async bulk GMEM→SMEM loads |
| FP4/FP8 Block Scaling | SM100 (B200) | `mxf8f6f4.block_scale` in MMA instruction |
| Tensor Memory (TMEM) | SM100 only | **Does not exist** |
| UMMA / tcgen05 | SM100 only | **Does not exist** |
| Shared Memory | Unique | **99 KB/SM** (vs 228 KB on SM100) |
| Cluster Multicast | SM100 | **Not supported** (1×1×1 only) |

The critical insight: **SM120 uses SM80-era `mma.sync` instructions, not SM100's `tcgen05.mma`.** Any kernel written for SM100 (including DeepGEMM, upstream CUTLASS SM100 collectives, and WGMMA-based Flash Attention) will fail to compile or crash at runtime on SM120.

### What this means in practice

- **`sm_120a` target rejects `tcgen05.*` instructions** at the ptxas level
- **`sm_100f` cubins fail to load** on SM120 hardware (driver rejects binary)
- SM120 GEMM tiles must fit **99 KB shared memory** (not SM100's 228 KB)
- No tensor memory means no UMMA descriptors — all data goes SMEM → registers → MMA

## Patches

This repository contains the patches needed to make vLLM + FlashInfer + CUTLASS run NVFP4 inference correctly on SM120. Each patch addresses a specific gap in upstream SM120 support.

### Core GEMM Patches

| # | Patch | What it fixes |
|---|-------|---------------|
| 1 | [FlashInfer SM120 Grouped GEMM](patches/flashinfer/01-sm120-grouped-gemm-tiles.md) | MoE grouped GEMM tile sizes restricted to 99 KB. Without this, M128/M256 tiles crash on SM120. |
| 2 | [QUTLASS SM120 Matmul](patches/vllm/02-qutlass-ada-mxf4-matmul.md) | SM120-optimized MXFP4 dense matmul using warp-level MMA + ldmatrix. |
| 3 | [CutlassExpertsFp4 SM120](patches/vllm/03-cutlass-experts-fp4-sm120.md) | Adds `is_device_capability_family(120)` to FP4 MoE support check. |
| 4 | [FlashInferExperts SM120](patches/vllm/04-flashinfer-experts-sm120.md) | Adds `is_device_capability(120)` to FlashInfer CUTLASS MoE dispatch. |

### Attention Patches

| # | Patch | What it fixes |
|---|-------|---------------|
| 5 | [Flash Attention SM120](patches/vllm/05-flash-attention-sm120.md) | SM120 subclass using SM80 MMA with 99 KB SMEM tile sizing. |
| 6 | [FlashInfer TMA FMHA SM120](patches/flashinfer/06-tma-fmha-sm120.md) | Pre-built FMHA kernels for SM120 (64×128, 64×64 tiles). |
| 7 | [FlashInfer MLA XQA SM120](patches/flashinfer/07-mla-xqa-sm120.md) | SM120 uses `xqa` backend for MLA (not `trtllm-gen`). |

### Infrastructure Patches

| # | Patch | What it fixes |
|---|-------|---------------|
| 8 | [FlashInfer JIT NVCC Flags](patches/flashinfer/08-jit-nvcc-sm120.md) | `sm120a` and `sm120f` gencode flags for JIT compilation. |
| 9 | [FlashInfer FP4 Quantization JIT](patches/flashinfer/09-fp4-quantization-sm120.md) | SM120 target for runtime activation quantization kernels. |
| 10 | [PyTorch Inductor Fixes](patches/vllm/10-inductor-monkeypatches.md) | 7 bug fixes for piecewise CUDA graph + NVFP4. |
| 11 | [quack SM120 GEMM](patches/quack/11-quack-gemm-sm120.md) | Custom warp-level GEMM (BF16/FP16, based on CUTLASS example 79). |
| 12 | [Marlin W4A8-FP8 SM120](patches/vllm/12-marlin-w4a8-sm120.md) | Enables Marlin FP8 kernels on SM120. |

## Benchmark Results

Single GPU, single request, `max_tokens=256`:

| Model | Format | tok/s | GPU |
|-------|--------|-------|-----|
| Qwen3.6-35B-A3B MoE | NVFP4 | **175** | RTX PRO 6000 ×1 |
| Gemma4-26B-A4B MoE | NVFP4 | 160 | RTX PRO 6000 ×1 |
| Qwen3.5-27B Dense | NVFP4 | 57 | RTX PRO 6000 ×1 |
| Gemma4-31B Dense | NVFP4 | 51 | RTX PRO 6000 ×1 |

## Architecture Deep Dive

See [docs/sm120-architecture.md](docs/sm120-architecture.md) for the full technical analysis, including:

- ISA comparison: SM80 vs SM90 vs SM100 vs SM120
- Shared memory capacity and tile size constraints
- MMA instruction encoding and register layout
- Why DeepGEMM (tcgen05-based) cannot work on SM120
- CUTLASS SM120 collective builder internals

## Quantization

The NVFP4 models linked above were quantized using [llm-compressor](https://github.com/vllm-project/llm-compressor) with the same patched environment:

```yaml
# recipe.yaml
default_stage:
  default_modifiers:
    QuantizationModifier:
      targets: [Linear]
      ignore: [lm_head, 're:.*visual.*', 're:.*mlp.gate$', 're:.*mlp.shared_expert_gate$']
      scheme: NVFP4
```

Key: quantize in the same CUDA 13.0 + patched environment to ensure the FP4 weight layout matches the SM120 kernel expectations.

## Requirements

- GPU: NVIDIA RTX 5090, 5080, 5070 Ti, RTX PRO 6000, or any SM120/SM121 GPU
- CUDA: 13.0+
- Driver: 580+
- Docker recommended (pre-built container: `lna-lab/gemma4-inference:latest`)

## Related Work

- [CUTLASS examples/87_blackwell_geforce_gemm_blockwise](https://github.com/NVIDIA/cutlass) — NVIDIA's official SM120 GEMM example
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) — SM100-only FP8 GEMM kernels (incompatible with SM120)
- [vLLM](https://github.com/vllm-project/vllm) — Upstream inference engine
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) — Attention and MoE kernels

## License

Apache-2.0

---

*Built at [Lna-Lab](https://github.com/lna-lab) — a seaside personal AI research lab.*
