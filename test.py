import cv2
from ultralytics import YOLO
from pathlib import Path

PROJECT_DIR = Path(r"H:\aiml learning\Projects\CNN\Vehicle detection")
TRAIN_IMAGE_DIR = PROJECT_DIR / "train" / "images"
OUTPUT_DIR = PROJECT_DIR / "results" / "train"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set the image path
image_paths = sorted(TRAIN_IMAGE_DIR.glob("*.jpg"))[:1]

print(f"Found {len(image_paths)} images")

# Load a pretrained YOLO model
model = YOLO("yolov8n.pt")
vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

total_vehicles = 0
total_type_counts = {name: 0 for name in vehicle_classes.values()}

# Load the image
for i, image_path in enumerate(image_paths, start=1):

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read", image_path.name, "Continuing")
        continue

    # Detect objects
    results = model.predict(image, conf=0.35, verbose=False)

    vehicle_count = 0
    type_counts = {name: 0 for name in vehicle_classes.values()}

    # Process detections
    for result in results:
        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Keep only vehicles
            if class_id not in vehicle_classes:
                continue

            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = vehicle_classes[class_id]
            text = f"{label} {confidence:.2f}"

            # Draw bounding box
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Add class label and confidence
            cv2.putText(
                image,
                text,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            vehicle_count += 1
            type_counts[label] +=1
            total_type_counts[label] +=1

    total_vehicles += vehicle_count

    # Save the output image
    output_path = OUTPUT_DIR / f"result_{image_path.name}"
    cv2.imwrite(str(output_path), image)

    counts_text = ", ".join(f"{name}: {count}" for name, count in type_counts.items())
    print(f"[{i}/{len(image_paths)}] {image_path.name}: {vehicle_count} vehicles ({counts_text})")

# Display the image
print("\nDone!")
print("Images processed:", len(image_paths))
print("Total vehicles detected:", total_vehicles)
print("Vehicles by type: ")
for name, count in total_type_counts.items():
    print(f" {name}: {count}")
print("Results saved in:", OUTPUT_DIR)