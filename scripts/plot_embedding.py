import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import os

def plot_tsne():
    # 1. Load data
    embed_path = "features/cnn_embeddings.csv"
    probs_path = "features/cnn_fold_probs.csv"
    
    if not os.path.exists(embed_path):
        print("File embedding gak ketemu! Cek folder features/")
        return

    df_embed = pd.read_csv(embed_path)
    df_probs = pd.read_csv(probs_path)

    # Ambil kolom fitur (cnn_embed_0 sampai cnn_embed_127)
    features = df_embed.filter(like='cnn_embed_').values
    labels = df_probs.sort_values("patient_id")["brugada"].values
    
    # Filter out NaN (buat jaga-jaga kalau ada failed subjects)
    mask = ~np.isnan(features).any(axis=1)
    features = features[mask]
    labels = labels[mask]

    print(f"Processing {len(features)} subjects...")

    # 2. Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    embeds_2d = tsne.fit_transform(features)

    # 3. Plotting
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x=embeds_2d[:, 0], y=embeds_2d[:, 1],
        hue=labels, palette={0: 'blue', 1: 'red'},
        alpha=0.7, edgecolor='w', s=100
    )
    
    plt.title("t-SNE Projection of Brugada ECG Embeddings", fontsize=15)
    plt.legend(title='Class', labels=['Normal (0)', 'Brugada (1)'])
    plt.grid(True, linestyle='--', alpha=0.6)
    
    save_path = "results/embeddings_tsne.png"
    plt.savefig(save_path)
    print(f"Grafik tersimpan di: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_tsne()