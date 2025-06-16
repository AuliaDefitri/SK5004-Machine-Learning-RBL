import pandas as pd

# 1. Nilai target Average_D dari gambar
average_d = [
    410, 80, 160, 394, 582, 490, 274, 330, 399, 581,
    228, 109, 500, 405, 383, 462, 217, 144, 136, 128,
    393, 568, 658, 150, 135, 240, 306, 292, 407, 1400,
    618, 375, 488, 267, 257, 196, 300, 223, 235, 412,
    387, 381, 405, 410, 438, 500, 350, 912, 400, 977,
    380, 297, 294, 250, 490, 432, 240, 490, 512, 516
]

# 2. Baca hasil PCA
pca_df = pd.read_csv('/content/drive/MyDrive/Machine Learning/RBL/COBA/features_pca.csv')

# 3. Periksa apakah jumlah baris sama
assert len(pca_df) == len(average_d), "Jumlah data PCA dan target Average_D tidak sama!"

# 4. Tambahkan kolom Average_D
pca_df['Average_D'] = average_d

# 5. Simpan sebagai file baru
pca_df.to_csv('/content/drive/MyDrive/Machine Learning/RBL/COBA/features_target.csv', index=False)

print("Berhasil digabung! Disimpan sebagai 'sem_image_features_pca90_with_target.csv'")
