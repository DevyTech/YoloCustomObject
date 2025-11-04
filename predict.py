from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best_ncnn_model")

# model.predict(source="test2.mp4", show=True, save=True)
model.predict(source="train/images/", show=True, save=True)