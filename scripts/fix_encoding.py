# scripts/fix_encoding.py
"""
Fix semua file write di interpretability modules
untuk menggunakan utf-8 encoding.
"""

import os
import re
from pathlib import Path

# Semua file yang perlu difix
target_files = [
    "experiments/interpretability/clinical_report.py",
    "experiments/interpretability/error_analysis.py",
    "experiments/interpretability/shap_analysis.py",
    "experiments/interpretability/permutation_importance.py",
    "experiments/interpretability/lead_analysis.py",
]

pattern = re.compile(r"open\(([^)]+)\s*,\s*'w'\s*\)")
replacement = r"open(\1, 'w', encoding='utf-8')"

fixed_count = 0
for fpath in target_files:
    if not Path(fpath).exists():
        print(f"SKIP (not found): {fpath}")
        continue

    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Cari semua open(..., 'w') tanpa encoding
    matches = pattern.findall(content)
    if not matches:
        print(f"OK (no fix needed): {fpath}")
        continue

    new_content = pattern.sub(replacement, content)

    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"FIXED ({len(matches)} occurrences): {fpath}")
    fixed_count += len(matches)

print(f"\nTotal fixes applied: {fixed_count}")