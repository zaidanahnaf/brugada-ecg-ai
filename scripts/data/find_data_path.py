# scripts/find_data_path.py

import os
from pathlib import Path
from collections import Counter


def main():
    search_root = "data"
    print(f"Searching under: {Path(search_root).resolve()}\n")

    hea_files = []
    csv_files = []

    for root, dirs, files in os.walk(search_root):
        for f in files:
            full = Path(root) / f
            if f.endswith(".hea"):
                hea_files.append(full)
            if f.endswith(".csv"):
                csv_files.append(full)

    # ── .hea files ───────────────────────────────────────────────
    print(f"Found {len(hea_files)} .hea files")
    if hea_files:
        counts   = Counter(f.parent for f in hea_files)
        best_dir = counts.most_common(1)[0][0]
        print(f"Most .hea files in: {best_dir}")
        print("\nFirst 5 .hea files:")
        for f in hea_files[:5]:
            print(f"  {f}")
    else:
        print("ERROR: No .hea files found at all.")
        print("Check your data folder structure:")
        for root, dirs, files in os.walk("data"):
            level = root.replace("data", "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            if level < 3:
                subindent = " " * 2 * (level + 1)
                for f in files[:3]:
                    print(f"{subindent}{f}")
        return

    # ── CSV / metadata ───────────────────────────────────────────
    print(f"\nFound {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  {f}")

    # ── Quick load test ───────────────────────────────────────────
    print("\n── Quick load test ──────────────────────────────────")
    try:
        import wfdb
        test_stem = hea_files[0].stem
        test_path = str(best_dir / test_stem)
        sig, fields = wfdb.rdsamp(test_path)
        print(f"  Record : {test_stem}")
        print(f"  Shape  : {sig.shape}")
        print(f"  FS     : {fields['fs']} Hz")
        print(f"  Leads  : {fields['sig_name']}")
        print("  LOAD TEST PASSED")
    except Exception as e:
        print(f"  LOAD TEST FAILED: {e}")

    # ── Recommended config.py fix ─────────────────────────────────
    print("\n" + "=" * 60)
    print("COPY THESE TWO LINES INTO src/config.py")
    print("=" * 60)
    print(f'\n  data_dir: str = "{best_dir.as_posix()}"')
    if csv_files:
        metadata_candidates = [f for f in csv_files if "metadata" in f.name.lower()]
        if metadata_candidates:
            print(f'  metadata_path: str = "{metadata_candidates[0].as_posix()}"')
        else:
            print(f'\n  No metadata.csv found. CSV files found:')
            for f in csv_files:
                print(f'    "{f.as_posix()}"')
    print("=" * 60)


if __name__ == "__main__":
    main()