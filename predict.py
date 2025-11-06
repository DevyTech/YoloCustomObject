from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best_ncnn_model")

# model.predict(source="test2.mp4", show=True, save=True)
# model.predict(source="train/images/", show=True, save=True)
result = model.predict(source="train/images/", show=True, save=True)
for r in result:
    print("Nama file:",r.path)
    print("Jumlah objek terdeteksi:",len(r.boxes))

    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()

        label = model.names[cls_id]
        print(f"Objek: {label}, Confidence: {conf:.2f}, Lokasi: {xyxy}")