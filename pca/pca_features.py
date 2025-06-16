import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load data fitur (hasil embedding dari InceptionV3)
features_df = pd.read_csv('/content/drive/MyDrive/Machine Learning/RBL/COBA/features.csv')

# Pisahkan nama file dan fitur
file_names = features_df['filename']
X = features_df.drop(columns=['filename']).values  # Hanya fitur (2048 dimensi)

# Standardisasi data (PCA sensitif terhadap skala)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Lakukan PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Hitung variasi kumulatif
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Cari jumlah komponen yang menyumbang ≥ 90% variasi
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
print(f"Jumlah komponen untuk mencapai ≥90% variasi: {n_components_90}")

# Transformasi ke dimensi baru dengan n_components_90
pca_90 = PCA(n_components=n_components_90)
X_pca_90 = pca_90.fit_transform(X_scaled)

# Buat DataFrame hasil PCA
pca_df = pd.DataFrame(X_pca_90)
pca_df.insert(0, "filename", file_names)

# Simpan hasil PCA ke CSV
pca_df.to_csv('/content/drive/MyDrive/Machine Learning/RBL/COBA/features_pca.csv', index=False)

# (Opsional) Plot variasi kumulatif
plt.figure(figsize=(8,5))
plt.plot(cumulative_variance, marker='o')
plt.axhline(y=0.9, color='r', linestyle='--', label='90% Variance')
plt.axvline(x=n_components_90-1, color='g', linestyle='--', label=f'{n_components_90} Components')
plt.title('PCA - Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance')
plt.legend()
plt.grid(True)
plt.show()

