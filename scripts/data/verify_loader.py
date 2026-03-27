# scripts/verify_loader.py

from src.data.data_loader import load_record
from src.config.__init__ import CFG

print(f"data_dir: {CFG.data.data_dir}")
print(f"metadata_path: {CFG.data.metadata_path}\n")

# Test 3 subjek pertama
test_ids = ["1009874", "1013884", "1023925"]

for pid in test_ids:
    result = load_record(pid)
    if result['valid']:
        print(f"OK  {pid}: shape={result['signal'].shape}, fs={result['fs']}, leads={result['lead_names']}")
    else:
        print(f"FAIL {pid}: {result['failure_reason']}")