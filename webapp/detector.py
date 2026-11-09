from pathlib import Path

import cv2
from ultralytics import YOLO

from timing import mark_front_row

PROJECT_DIR = Path(__file__).parent.parent
MODEL_PATH = PROJECT_DIR / "tuning" / "yolo11_vehicle_v1" / "weights" / "best.pt"
CONFIDENCE = 0.35

# Box colour for each class (BGR)
CLASS_COLORS = {
    "car": (0, 200, 0),
    "motorcycle": (255, 140, 0),
    "bus": (0, 140, 255),
    "truck": (200, 0, 200),
}
IGNORED_COLOR = (150, 150, 150)  # 2-wheelers behind the front row

# Load the fine-tuned model once when the app starts, not on every request
model = YOLO(str(MODEL_PATH))


def detect_vehicles(image_path, output_path):
    """Detect vehicles, mark the front row, save a copy with boxes drawn, and return the detections."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    results = model.predict(image, conf=CONFIDENCE, verbose=False)

    counts = {name: 0 for name in model.names.values()}
    detections = []

    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls[0])]
            counts[label] += 1
            detections.append({
                "label": label,
                "confidence": float(box.conf[0]),
                "box": tuple(map(int, box.xyxy[0])),
            })

    mark_front_row(detections)

    for d in detections:
        x1, y1, x2, y2 = d["box"]
        ignored = d["front_row"] is False  # only small vehicles get True/False
        color = IGNORED_COLOR if ignored else CLASS_COLORS.get(d["label"], (0, 255, 0))
        thickness = 3 if d["front_row"] else 2

        tag = "FRONT" if d["front_row"] else ("ignored" if ignored else "")
        text = f"{d['label']} {d['confidence']:.2f}" + (f" {tag}" if tag else "")

        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(image, text, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite(str(output_path), image)

    return {"counts": counts, "total": len(detections), "detections": detections}
