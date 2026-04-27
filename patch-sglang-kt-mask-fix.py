"""
Patch mask_cpu_expert_ids and apply method for CUTLASS compatibility.
Uses string-exact replacement instead of regex to avoid docstring issues.
"""

FILE = "/sgl-workspace/sglang/python/sglang/srt/layers/moe/kt_ep_wrapper.py"

with open(FILE) as f:
    content = f.read()

# Fix 1: In the apply method, use 0 + zeroed weights instead of -1
OLD_APPLY = """        # Step 2: Prepare GPU computation by masking CPU expert IDs
        # CPU expert IDs (>= num_gpu_experts) are set to -1 so GPU kernel skips them
        topk_ids = topk_output.topk_ids
        masked_topk_ids = mask_cpu_expert_ids(topk_ids, self.num_gpu_experts)

        # Create modified dispatch output for GPU computation
        masked_topk_output = topk_output._replace(topk_ids=masked_topk_ids)"""

NEW_APPLY = """        # Step 2: Prepare GPU computation by masking CPU expert IDs
        # CPU expert IDs (>= num_gpu_experts) set to 0, weights zeroed (CUTLASS compat)
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights
        cpu_mask = topk_ids >= self.num_gpu_experts
        masked_topk_ids = topk_ids.clone()
        masked_topk_ids[cpu_mask] = 0
        masked_topk_weights = topk_weights.clone()
        masked_topk_weights[cpu_mask] = 0.0

        # Create modified dispatch output for GPU computation
        masked_topk_output = topk_output._replace(topk_ids=masked_topk_ids, topk_weights=masked_topk_weights)"""

assert OLD_APPLY in content, f"Cannot find apply mask section"
content = content.replace(OLD_APPLY, NEW_APPLY)

with open(FILE, "w") as f:
    f.write(content)

print("Patched: apply() uses 0 + zeroed weights instead of -1 for CPU experts.")
