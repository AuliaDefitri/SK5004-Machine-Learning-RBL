# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
!pip install tensorflow opencv-python

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

# Path folder hasil gambar
input_folder = '/content/drive/MyDrive/Machine Learning/RBL/COBA/sem_images_result'

# Load pre-trained InceptionV3 model (tanpa top/classification layer)
base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')  # Output: 2048-dim

# List untuk menyimpan data
feature_list = []
file_names = []

# Target size untuk InceptionV3
iv3_target_size = (299, 299)  # width x height

# Loop semua gambar hasil
for filename in tqdm(sorted(os.listdir(input_folder))):
    if filename.lower().endswith(".png") and '_Result' in filename:
        path = os.path.join(input_folder, filename)

        # Baca gambar dan konversi ke RGB
        image = cv2.imread(path)
        if image is None:
            print(f"Skipped (failed read): {filename}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, iv3_target_size)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        # Ekstraksi fitur
        features = base_model.predict(image)
        features = features.flatten()  # Ubah jadi 1D array

        # Simpan fitur
        feature_list.append(features)
        file_names.append(filename)

# Konversi ke DataFrame
features_df = pd.DataFrame(feature_list)
features_df.insert(0, "filename", file_names)

# Simpan sebagai CSV
features_df.to_csv('/content/drive/MyDrive/Machine Learning/RBL/COBA/features.csv', index=False)

print("Embedding selesai. Fitur disimpan sebagai features.csv")
