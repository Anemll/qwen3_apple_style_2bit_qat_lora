#!/usr/bin/env python3
"""
Copy rank_magnitude tensors from unsnapped to snapped checkpoint.
Tests if FP16 rounding of rank_magnitude is causing divergence.
"""

import torch

# Paths
unsnapped = "/Users/anemll/Downloads/SR-008-32B/checkpoint_STE_200.pt"
snapped   = "/Users/anemll/Downloads/SR-008-32B/checkpoint_STE_200_snapped1.pt"
output    = "/Users/anemll/Downloads/SR-008-32B/checkpoint_STE_200_snapped1_rmfixed.pt"

print(f"Source (unsnapped): {unsnapped}")
print(f"Target (snapped):   {snapped}")
print(f"Output:             {output}")

# Load
print("\nLoading checkpoints...")
src_sd = torch.load(unsnapped, map_location="cpu")
tgt_sd = torch.load(snapped, map_location="cpu")

# Handle nested state_dict
if isinstance(src_sd, dict) and "model_state_dict" in src_sd:
    src_sd = src_sd["model_state_dict"]
if isinstance(tgt_sd, dict) and "model_state_dict" in tgt_sd:
    tgt_sd = tgt_sd["model_state_dict"]

# Copy rank_magnitude from source to target
copied = 0
max_diff = 0.0

for k in list(tgt_sd.keys()):
    if k.endswith(".rank_magnitude") and k in src_sd:
        src_val = src_sd[k]
        tgt_val = tgt_sd[k]

        # Compute difference
        diff = (src_val.float() - tgt_val.float()).abs()
        max_diff = max(max_diff, diff.max().item())

        # Replace with source value
        tgt_sd[k] = src_val.clone()
        copied += 1

print(f"\nCopied {copied} rank_magnitude tensors")
print(f"Max difference was: {max_diff:.6f}")

# Save
torch.save(tgt_sd, output)
print(f"\nSaved: {output}")
print(f"\nTest with:\npython scripts/test_inference.py {output} --lora-r 8")
