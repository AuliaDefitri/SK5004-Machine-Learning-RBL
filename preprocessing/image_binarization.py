!pip install opencv-python
import os
import cv2

# Folder input dan output di Google Drive
input_folder = '/content/drive/MyDrive/Machine Learning/RBL/COBA/sem_images'
output_folder = '/content/drive/MyDrive/Machine Learning/RBL/COBA/sem_images_result'

# Pastikan folder output ada
os.makedirs(output_folder, exist_ok=True)

# Ukuran target untuk resize (misalnya 256x256)
target_size = (256, 256)  # Format: (width, height)

# Proses setiap file PNG di folder input
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".png"):
        input_path = os.path.join(input_folder, filename)

        # Baca gambar dalam grayscale
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Gagal membaca gambar: {filename}")
            continue

        # Resize gambar ke ukuran target
        resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

        # Ubah ke citra biner menggunakan Otsu threshold
        _, binary_image = cv2.threshold(resized_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Buat nama file baru
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}_Result.png"
        output_path = os.path.join(output_folder, output_filename)

        # Simpan gambar biner
        cv2.imwrite(output_path, binary_image)

        print(f"Processed: {filename} → {output_filename}")

print("Semua gambar telah diproses, di-resize, dan disimpan di folder sem_images_result.")
