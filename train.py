import yaml
import os

from ultralytics import YOLO
import torch

DATASET_PATH = r"H:\aiml learning\Projects\CNN\Vehicle detection"

data_config = {
    "path": DATASET_PATH,
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "names": {
        0: "car",
        1: "motorcycle",
        2: "bus",
        3: "truck"
    }
}

yaml_path = os.path.join(DATASET_PATH, "data.yaml")

with open(yaml_path, "w") as file:
    yaml.safe_dump(data_config, file, sort_keys=False)

print("Dataset configuration saved:", yaml_path)

# Check whether a GPU is available
device = 0 if torch.cuda.is_available() else "cpu"
print("Training device:", device)

# Load pretrained YOLO11 nano model
model = YOLO("yolo11n.pt")

# Fine-tune on your vehicle dataset
results = model.train(
    data=yaml_path,
    epochs=100,
    imgsz=640,
    batch=-1,
    patience=20,

    # Optimization
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,
    cos_lr=True,

    # Data augmentation
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.4,
    degrees=5.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    close_mosaic=10,

    # Reproducibility and output
    seed=42,
    pretrained=True,
    plots=True,
    save=True,
    workers=2,
    device=device,
    project=r"H:\aiml learning\Projects\CNN\Vehicle detection\tuning",
    name="yolo11_vehicle_v1"
)

print("Training completed!")

best_model = YOLO(model.trainer.best)

metrics = best_model.val(
    data=yaml_path,
    split="val",
    imgsz=640,
    plots=True
)

print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)