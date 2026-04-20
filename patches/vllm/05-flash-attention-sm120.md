# Patch 5: Flash Attention SM120 Subclass

**Priority: High** — Enables optimized attention on SM120.

## Problem

Flash Attention's SM90/SM100 paths use WGMMA, which SM120 does not have. The SM80 path works but doesn't know about SM120's 99 KB shared memory capacity, defaulting to SM80's 163 KB budget.

## Location

- `vllm/vllm_flash_attn/cute/flash_fwd_sm120.py`
- `vllm/vllm_flash_attn/cute/flash_bwd_sm120.py`
- `vllm/vllm_flash_attn/cute/interface.py` (dispatch, lines ~475)

## Fix

New subclasses `FlashAttentionForwardSm120` and `FlashAttentionBackwardSm120` that:

1. Inherit from SM80 implementation (warp-level MMA)
2. Override `can_implement()` with SM120's 99 KB SMEM capacity
3. Select tile sizes tuned for 99 KB:

```python
# Forward:
if head_dim <= 64:
    fwd_cfg = FwdConfig(128, 128, True, True)  # 48 KB
else:
    fwd_cfg = FwdConfig(128, 64, True, True)   # 64 KB

# Backward:
m_block_size = 64
n_block_size = 64
num_stages_Q = 2 if head_dim <= 64 else 1
```

Dispatch gate: `arch // 10 == 12` → SM120/SM121.

## Design Decision

SM120 intentionally uses SM80's MMA instructions (`mma.sync.aligned.m16n8k16`) for attention. The `arch = 80` class attribute is kept to select the SM80 code path, while compilation targets `sm_120a`. This is correct: SM120's MMA is derived from SM80.

## Impact

Without: SM80 attention with wrong SMEM budget → either crash or poor occupancy
With: Correctly sized attention tiles → stable, good occupancy
