from ultralytics import YOLO
import cv2
import os
from collections import defaultdict
import time

VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

def analyze_traffic_flow(input_path, output_folder="results/detections"):
    if not os.path.exists(input_path):
        print(f"Error: File not found → {input_path}")
        return

    print("Loading model for Traffic Flow Analysis...")
    model = YOLO("yolov8n.pt")
    os.makedirs(output_folder, exist_ok=True)

    unique_vehicles = defaultdict(set)
    frame_vehicle_counts = []   # how many vehicles visible per frame

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps
    cap.release()

    print(f"Video FPS: {fps:.1f}")
    print(f"Duration: {duration_seconds:.1f} seconds")
    print("Processing...")

    results = model.track(
        source=input_path,
        conf=0.4,
        classes=list(VEHICLE_CLASSES.keys()),
        persist=True,
        stream=True,
        verbose=False
    )

    frame_count = 0

    for result in results:
        current_count = 0

        if result.boxes is not None and result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()
            class_ids = result.boxes.cls.int().cpu().tolist()

            for track_id, cls_id in zip(track_ids, class_ids):
                if cls_id in VEHICLE_CLASSES:
                    class_name = VEHICLE_CLASSES[cls_id]
                    unique_vehicles[class_name].add(track_id)
                    current_count += 1

        frame_vehicle_counts.append(current_count)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    # ========== CALCULATIONS ==========
    total_unique = sum(len(ids) for ids in unique_vehicles.values())
    avg_vehicles_on_screen = sum(frame_vehicle_counts) / len(frame_vehicle_counts) if frame_vehicle_counts else 0
    vehicles_per_minute = (total_unique / duration_seconds) * 60 if duration_seconds > 0 else 0

    # Simple density estimation
    if avg_vehicles_on_screen < 3:
        condition = "Free Flow"
    elif avg_vehicles_on_screen < 7:
        condition = "Moderate"
    elif avg_vehicles_on_screen < 12:
        condition = "Heavy"
    else:
        condition = "Congested"

    # ========== RESULTS ==========
    print("\n" + "="*40)
    print("       TRAFFIC FLOW ANALYSIS")
    print("="*40)
    print(f"Video Duration          : {duration_seconds:.1f} seconds")
    print(f"Total Unique Vehicles   : {total_unique}")
    print(f"Vehicles per Minute     : {vehicles_per_minute:.1f}")
    print(f"Avg Vehicles on Screen  : {avg_vehicles_on_screen:.1f}")
    print(f"Estimated Condition     : {condition}")
    print("-"*40)
    print("Breakdown by type:")
    for name, ids in unique_vehicles.items():
        print(f"  {name.capitalize():<12}: {len(ids)}")
    print("="*40)


if __name__ == "__main__":
    input_path = "data/raw_videos/sample_traffic.mp4"
    analyze_traffic_flow(input_path)