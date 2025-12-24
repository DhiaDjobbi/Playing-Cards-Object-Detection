from ultralytics import YOLO

# Load trained model
model = YOLO("final_models/yolov8m_tuned.pt")

# Export to INT8 TFLite
# data\real_dataset\data.yaml
model.export(
    format="tflite",
    int8=True,
    imgsz=640,
    data="data/real_dataset/data.yaml"
)
print("Model exported to INT8 TFLite format successfully.")