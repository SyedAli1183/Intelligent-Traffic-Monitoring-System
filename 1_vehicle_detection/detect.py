from ultralytics import YOLO
import cv2
import os
from collections import defaultdict

VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

def detect_track_count(input_path, output_folder="results/detections"):
    if not os.path.exists(input_path):
        print(f"Error: File not found → {input_path}")
        return

    print("Loading YOLOv8 model with tracking...")
    model = YOLO("yolov8n.pt")
    os.makedirs(output_folder, exist_ok=True)

    # Dictionary to store unique vehicle IDs
    unique_vehicles = defaultdict(set)   # class_name → set of track_ids

    print("Processing video with tracking... This may take a few minutes.")
    results = model.track(
        source=input_path,
        conf=0.4,
        classes=list(VEHICLE_CLASSES.keys()),  # only vehicles
        persist=True,
        stream=True,
        verbose=False
    )

    # Prepare video writer
    cap = cv2.VideoCapture(input_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    cap.release()

    output_path = os.path.join(output_folder, "tracked_and_counted.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    for result in results:
        frame = result.orig_img.copy()

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()
            class_ids = result.boxes.cls.int().cpu().tolist()
            confidences = result.boxes.conf.cpu().numpy()

            for box, track_id, cls_id, conf in zip(boxes, track_ids, class_ids, confidences):
                if cls_id not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box)
                class_name = VEHICLE_CLASSES[cls_id]

                # Store unique ID
                unique_vehicles[class_name].add(track_id)

                # Draw box and ID
                label = f"ID:{track_id} {class_name} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show current unique counts on frame
        y = 30
        for name, ids in unique_vehicles.items():
            text = f"{name}: {len(ids)}"
            cv2.putText(frame, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            y += 30

        out.write(frame)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    out.release()

    # Final Results
    print("\n===== Final Tracking Results =====")
    total = 0
    for name, ids in unique_vehicles.items():
        count = len(ids)
        total += count
        print(f"{name.capitalize()}: {count}")
    print(f"Total unique vehicles: {total}")
    print(f"Video saved to → {output_path}")


if __name__ == "__main__":
    input_path = "data/raw_videos/sample_traffic.mp4"
    detect_track_count(input_path)