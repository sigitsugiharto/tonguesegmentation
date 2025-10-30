from ultralytics import YOLO
model = YOLO("18best.pt")
source = "coba.jpg"
hasil = "hasil"
results = model.predict(source=source, save=True, project="hasil", name="nomor", imgsz=640)

from IPython.display import Image, display

results = model.predict(source=source, save=False)
for r in results:
    print(r.boxes.xyxy)  # koordinat bounding box
    print(r.masks.xy)    # koordinat polygon mask (kalau model segmentation)
    print(r.names)       # nama kelas

display(Image(filename=f"hasil/{hasil}/{source}"))
