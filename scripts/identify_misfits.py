import pandas as pd
import numpy as np
from sklearn.manifold import TSNE

def find_misfits():
    # 1. Load data
    df_embed = pd.read_csv("features/cnn_embeddings.csv")
    df_probs = pd.read_csv("features/cnn_fold_probs.csv")
    
    # Merge biar ID, Label, dan Embedding sinkron
    df = pd.merge(df_embed, df_probs, on="patient_id")
    df = df.dropna(subset=["oof_prob_brugada"])
    
    # Ambil fitur embedding
    features = df.filter(like='cnn_embed_').values
    
    # 2. Re-run t-SNE buat dapet koordinat 2D (sama kyk plot lo)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    coords = tsne.fit_transform(features)
    df['tsne_x'] = coords[:, 0]
    df['tsne_y'] = coords[:, 1]

    # 3. Cari Centroid (Pusat) masing-masing class di 2D
    center_normal = coords[df['brugada'] == 0].mean(axis=0)
    center_brugada = coords[df['brugada'] == 1].mean(axis=0)

    # Hitung jarak tiap titik ke pusat class-nya SENDIRI
    def get_dist(row):
        center = center_normal if row['brugada'] == 0 else center_brugada
        return np.linalg.norm([row['tsne_x'] - center[0], row['tsne_y'] - center[1]])

    df['dist_to_center'] = df.apply(get_dist, axis=1)

    # 4. Filter Misfits (Yang Prediksinya Salah)
    # False Negative (FN): Aslinya Brugada (1), tapi Probabilitas Rendah (< 0.5)
    fn = df[(df['brugada'] == 1) & (df['oof_prob_brugada'] < 0.5)].sort_values('dist_to_center', ascending=False)
    
    # False Positive (FP): Aslinya Normal (0), tapi Probabilitas Tinggi (> 0.5)
    fp = df[(df['brugada'] == 0) & (df['oof_prob_brugada'] > 0.5)].sort_values('dist_to_center', ascending=False)

    print("\n=== TOP 5 FALSE NEGATIVES (Brugada Missed) ===")
    print(fn[['patient_id', 'oof_prob_brugada', 'fold_id']].head(5))

    print("\n=== TOP 5 FALSE POSITIVES (Normal Misclassified) ===")
    print(fp[['patient_id', 'oof_prob_brugada', 'fold_id']].head(5))

if __name__ == "__main__":
    find_misfits()