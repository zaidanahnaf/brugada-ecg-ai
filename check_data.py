import wfdb
import pandas as pd
from pathlib import Path

# Path utama dataset
base_dir = Path("data/raw/brugada-huca-12-lead-ecg-recordings-for-the-study-of-brugada-syndrome-1.0.0")
# Path lokasi file metadata dan rekaman (.hea/.dat)
metadata_path = base_dir / "metadata.csv"
records_dir = base_dir / "files"

def run_validation():
    if not metadata_path.exists():
        print(f"Error: Metadata tidak ditemukan di {metadata_path}")
        return

    metadata = pd.read_csv(metadata_path)
    print(f"Metadata rows    : {len(metadata)}")
    print(f"Brugada positive : {metadata['brugada'].sum()}")
    print(f"Brugada negative : {(metadata['brugada'] == 0).sum()}")
    print("-" * 30)

    print("Spot checking first 5 records:")
    for pid in metadata['patient_id'].astype(str).head(5):
        # Gabungkan path folder 'files' dengan ID pasien
        record_path = str(records_dir / pid)
        
        try:
            # wfdb.rdsamp akan otomatis mencari {pid}.hea dan {pid}.dat di folder 'files'
            sig, fields = wfdb.rdsamp(record_path)
            print(f"  {pid}: shape={sig.shape}, fs={fields['fs']}, leads={fields['sig_name']}")
        except Exception as e:
            print(f"  {pid}: FAILED — {e}")

if __name__ == "__main__":
    run_validation()
