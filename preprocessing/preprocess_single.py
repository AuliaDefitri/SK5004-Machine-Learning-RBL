import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, exposure
from skimage.morphology import opening, closing, disk
from tensorflow.keras.preprocessing.image import img_to_array
import os

def preprocess_image(image_path, target_size=(224, 224)):
    # 1. Baca gambar dan konversi ke grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Segmentasi dengan Otsu's Thresholding
    thresh = filters.threshold_otsu(gray)
    binary = (gray > thresh).astype(np.uint8) * 255

    # 3. Operasi Morfologi (Membersihkan noise)
    kernel = disk(3)  # Kernel lingkaran radius 3
    cleaned = opening(binary, kernel)  # Menghilangkan noise kecil

    # 4. Temukan kontur terbesar (Asumsi: objek utama adalah kontur terbesar)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    # 5. Buat mask dari kontur
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # 6. Isolasi objek dengan mask
    segmented = cv2.bitwise_and(img, img, mask=mask)

    # 7. Normalisasi intensitas piksel
    normalized = exposure.rescale_intensity(segmented, out_range=(0, 255))

    # 8. Resize ke ukuran yang diinginkan (untuk VGG16)
    resized = cv2.resize(normalized, target_size)

    # 9. Konversi ke RGB (VGG16 membutuhkan 3 channel)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    return rgb
