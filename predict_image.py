from ultralytics import YOLO
import sys
from pathlib import Path

# ===== CONFIG =====
MODEL_PATH = "final_models/yolov8m_tuned.pt"  # change if needed
CONF_THRESHOLD = 0.5
# ==================

def predict_image(image_path: str):
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return

    model = YOLO(MODEL_PATH)

    results = model(
        source=image_path,
        conf=CONF_THRESHOLD,
        device="cpu",
        verbose=False
    )

    detections = results[0]

    if detections.boxes is None:
        print("No cards detected.")
        return

    print(f"\n📷 Image: {image_path}")
    print("🃏 Detected cards:")

    for cls_id, conf in zip(detections.boxes.cls, detections.boxes.conf):
        class_name = detections.names[int(cls_id)]
        confidence = float(conf)

        print(f"  - {class_name} ({confidence:.2f})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_image.py <image.jpg>")
        sys.exit(1)

    predict_image(sys.argv[1])
