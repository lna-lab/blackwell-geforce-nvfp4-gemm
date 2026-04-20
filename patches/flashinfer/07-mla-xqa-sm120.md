# Patch 7: FlashInfer MLA XQA for SM120

**Priority: Low** — Only needed for models using MLA (Multi-Latent Attention, e.g. DeepSeek).

## Fix

SM120 uses the `xqa` (cross-query-attention) backend for MLA, not `trtllm-gen`.

Location: `flashinfer/mla.py` (lines ~643-686)
XQA kernel: `flashinfer/jit/xqa.py` → `xqa/mla_sm120.cu`
