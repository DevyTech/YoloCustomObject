from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train/weights/best_ncnn_model")

# model.predict(source="test2.mp4", show=True, save=True)
# model.predict(source="train/images/", show=True, save=True)
# result = model.predict(source="test.mp4", show=True, save=True)
result = model.predict(source="2.mp4", stream=True)
for r in result:
    frame = r.orig_img.copy()
    boxes = r.boxes
    rokok_detected = False  # flag deteksi
    print("Nama file:",r.path)
    print("Jumlah objek terdeteksi:",len(boxes))

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        label = model.names[cls_id]

        # Ambil koordinat
        x1, y1, x2, y2 = map(int,xyxy)
        color = (0,255,0) if label != "Rokok" else (0,0,255)

        # Gambar kotak deteksi
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        text = f"{label} ({conf:.2f})"
        cv2.putText(frame, text, (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

        if label.lower() == "rokok":
            rokok_detected = True


    if rokok_detected:
        cv2.putText(frame,"Rokok Detected!",(50,50),cv2.FONT_HERSHEY_DUPLEX,1.0,(0,0,255),3)
    else:
        cv2.putText(frame,"Rokok Undetected",(50,50),cv2.FONT_HERSHEY_DUPLEX,1.0,(0,255,0),2)


    cv2.imshow("Hasil Deteksi", frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cv2.destroyAllWindows()
        # print(f"Objek: {label}, Confidence: {conf:.2f}, Lokasi: {xyxy}")
        # if label == "Rokok":
        #     print("Rokok Detected")
        # else:
        #     print("Rokok Undetected")
