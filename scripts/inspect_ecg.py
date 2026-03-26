import wfdb
import matplotlib.pyplot as plt
import os
import glob

def plot_misfit(patient_id, root_dir="data/raw"):
    # Cari file .hea (wfdb butuh path ke header tanpa extension)
    search_pattern = os.path.join(root_dir, "**", str(patient_id), f"{patient_id}.hea")
    files = glob.glob(search_pattern, recursive=True)
    
    if not files:
        print(f"❌ File .hea buat ID {patient_id} gak ketemu!")
        return

    # Ambil path lengkap TANPA ekstensi .hea
    file_record_path = files[0].replace('.hea', '')
    print(f"✅ Nemukan record: {file_record_path}")

    try:
        # Baca record pakai wfdb
        record = wfdb.rdrecord(file_record_path)
        signals = record.p_signal.T  # Transpose supaya (channels, samples)
        
        # Lead V1, V2, V3 (Biasanya index 6, 7, 8 di 12-lead standard)
        # Tapi lebih aman cek record.sig_name kalau indexnya beda
        indices = [6, 7, 8]
        names = ["V1", "V2", "V3"]
        
        plt.figure(figsize=(12, 10))
        for i, idx in enumerate(indices):
            plt.subplot(3, 1, i+1)
            # Ambil 1000 sample (2 detik jika 500Hz)
            segment = signals[idx, :1000]
            plt.plot(segment, color='crimson' if i==0 else 'black', lw=1.2)
            plt.title(f"Patient {patient_id} - Lead {record.sig_name[idx]}")
            plt.ylabel(f"Amplitude ({record.units[idx]})")
            plt.grid(True, alpha=0.3, linestyle='--')
            
        plt.tight_layout()
        save_name = f'results/inspeksi_{patient_id}.png'
        plt.savefig(save_name)
        print(f"💾 Plot berhasil disimpan sebagai: {save_name}")
        plt.show()

    except Exception as e:
        print(f"Error pas buka file pakai wfdb: {e}")

if __name__ == "__main__":
    plot_misfit("419960", root_dir="data/raw")