# scripts/audit_model_params.py
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

# Load model checkpoint langsung — tidak perlu configs
model_path = 'models/cnn_fold_0.pt'
state_dict = torch.load(model_path, map_location='cpu')

print("=== LAYER NAMES & SHAPES ===")
total_params = 0
for key, tensor in state_dict.items():
    params = tensor.numel()
    total_params += params
    print(f"  {key:<55} {str(tensor.shape):<25} {params:>10,}")

print(f"\n{'='*60}")
print(f"Total parameters: {total_params:,}")