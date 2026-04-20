# Patch 10: PyTorch Inductor Bug Fixes

**Priority: Critical** — Without these, CUDA graph compilation crashes.

## Problem

vLLM's `piecewise` CUDA graph mode (required for SM120 NVFP4 MoE) triggers several bugs in PyTorch's Inductor compiler (2.9.x–2.11.x). These are upstream bugs, not SM120-specific, but they surface specifically in the NVFP4 + piecewise CUDA graph combination.

## Location

`vllm/env_override.py` — executed on `import vllm`

## Fixes (7 monkeypatches)

1. **`memory_plan_reuse_patched`** — Piecewise CUDA graph compilation fails a test assertion. (PyTorch PR #165514)

2. **`get_graph_partition_signature_patched`** — Inductor partition + attention-NVFP4 quant fusion bug causes test_attn_quant failure. Critical for Gemma4 attention path. (PyTorch PR #165815)

3. **`should_partition_patched`** — Inductor scheduler crash when `use_inductor_graph_partition=True` with in-place mutation operators like `vllm.unified_attention_with_output`. (vLLM issue #26678)

4. **`_update_scheduler_patched`** — Related scheduler fix for the same in-place mutation issue.

5. **`_patch_get_raw_stream_if_needed`** — `get_raw_stream()` undefined during autotune. (vLLM issue #30905)

6. **`_apply_constrain_to_fx_strides_patch`** — Lowering crash on FakeScriptObject. (PyTorch issue #175973)

7. **`_patched_get_runtime_env`** — `GraphCaptureOutput.get_runtime_env` missing builtins. (PyTorch PR #177558)

## Why piecewise mode is required

SM120's NVFP4 MoE path uses operator fusion (norm_quant, act_quant) that requires Inductor compilation. The `full` CUDA graph mode can't capture the dynamic MoE routing. `piecewise` mode captures static subgraphs and leaves dynamic parts eager — but it triggers all these Inductor bugs.

## Impact

Without: `VLLM_CUDA_GRAPH_MODE=piecewise` crashes → must run in eager mode → **~8× slower**
With: Piecewise CUDA graphs work → 175 tok/s
