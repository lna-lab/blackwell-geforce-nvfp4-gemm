"""
Patch SGLang kt_ep_wrapper.py for kt-kernel 0.5.1 API compatibility.

Problem: SGLang passes num_gpu_experts= to KTMoEWrapper.__new__(),
but kt-kernel 0.5.1 expects gpu_experts_mask= (a tensor mask).

Fix: Convert num_gpu_experts to gpu_experts_mask before passing.
"""

FILE = "/sgl-workspace/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py"

with open(FILE) as f:
    content = f.read()

OLD = """            self.wrapper = KTMoEWrapper(
                layer_idx=self.kt_config.layer_idx,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                hidden_size=hidden_size,
                moe_intermediate_size=intermediate_size_full,
                num_gpu_experts=self.num_gpu_experts,
                cpuinfer_threads=self.kt_config.cpuinfer_threads,
                threadpool_count=self.kt_config.threadpool_count,
                weight_path=self.kt_config.weight_path,
                chunked_prefill_size=self.kt_config.chunked_prefill_size,
                method=self.kt_config.method,
                max_deferred_experts_per_token=layer_max_deferred,
            )"""

NEW = """            # Build gpu_experts_mask: True for GPU experts (0..num_gpu_experts-1)
            import torch as _torch
            _gpu_mask = _torch.zeros(num_experts, dtype=_torch.bool)
            _gpu_mask[:self.num_gpu_experts] = True
            self.wrapper = KTMoEWrapper(
                layer_idx=self.kt_config.layer_idx,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                hidden_size=hidden_size,
                moe_intermediate_size=intermediate_size_full,
                gpu_experts_mask=_gpu_mask,
                cpuinfer_threads=self.kt_config.cpuinfer_threads,
                threadpool_count=self.kt_config.threadpool_count,
                weight_path=self.kt_config.weight_path,
                chunked_prefill_size=self.kt_config.chunked_prefill_size,
                method=self.kt_config.method,
                max_deferred_experts_per_token=layer_max_deferred,
            )"""

assert OLD in content, "Cannot find KTMoEWrapper constructor call"
content = content.replace(OLD, NEW)

with open(FILE, "w") as f:
    f.write(content)

print("Patched: num_gpu_experts → gpu_experts_mask for kt-kernel 0.5.1 compat.")
