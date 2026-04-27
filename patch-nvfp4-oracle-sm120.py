"""
Patch vLLM oracle/nvfp4.py for SM120:
Reorder AVAILABLE_BACKENDS to skip SM100-only FlashInfer backends.
VLLM_CUTLASS (patched CutlassExpertsFp4) works on SM120.
"""
import re

FILE = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py"

with open(FILE) as f:
    content = f.read()

OLD = """    AVAILABLE_BACKENDS = [
        NvFp4MoeBackend.FLASHINFER_TRTLLM,
        NvFp4MoeBackend.FLASHINFER_CUTEDSL,
        NvFp4MoeBackend.FLASHINFER_CUTEDSL_BATCHED,
        NvFp4MoeBackend.FLASHINFER_CUTLASS,
        NvFp4MoeBackend.VLLM_CUTLASS,
        NvFp4MoeBackend.MARLIN,
    ]"""

NEW = """    # SM120 patch: skip FlashInfer backends (SM100-only CUDA kernels).
    # VLLM_CUTLASS uses CutlassExpertsFp4 which is patched for SM120.
    AVAILABLE_BACKENDS = [
        NvFp4MoeBackend.VLLM_CUTLASS,
        NvFp4MoeBackend.MARLIN,
    ]"""

assert OLD in content, "Pattern not found — vLLM version mismatch?"
content = content.replace(OLD, NEW)

with open(FILE, "w") as f:
    f.write(content)

print("Patched oracle/nvfp4.py: VLLM_CUTLASS first, SM100 backends removed.")
