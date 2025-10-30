from ultralytics import YOLO
import cv2
import os

model = YOLO("18best.pt")

# gambar uji
source = "coba.jpg"

# jalankan prediksi
results = model.predict(source=source, save=True, imgsz=640)

import numpy as np

# buat folder output crop
os.makedirs("crops", exist_ok=True)

for r in results:
    img = cv2.imread(source)
    boxes = r.boxes.xyxy  # [x1, y1, x2, y2] per objek
    names = r.names
    cls = r.boxes.cls  # ID kelas

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        crop = img[y1:y2, x1:x2]  # potong gambar
        label = names[int(cls[i])]
        path = f"crops/{label}_{i}.jpg"
        cv2.imwrite(path, crop)
        print(f"✅ disimpan: {path}")

import numpy as np

os.makedirs("mask_crops", exist_ok=True)

for r in results:
    img = cv2.imread(source)
    names = r.names
    cls = r.boxes.cls

    if r.masks is None:
        print("❌ Tidak ada mask ditemukan.")
        continue

    for i, mask in enumerate(r.masks.data):
        mask_np = mask.cpu().numpy().astype(np.uint8) * 255  # konversi ke array 0–255
        masked_img = cv2.bitwise_and(img, img, mask=mask_np)
        x1, y1, x2, y2 = map(int, r.boxes.xyxy[i])
        crop = masked_img[y1:y2, x1:x2]

        label = names[int(cls[i])]
        path = f"mask_crops/{label}_{i}.jpg"
        cv2.imwrite(path, crop)
        print(f"✅ crop mask disimpan: {path}")

from IPython.display import Image, display
display(Image(filename='mask_crops/tongue_0.jpg'))
