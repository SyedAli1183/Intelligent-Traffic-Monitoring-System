import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from collections import defaultdict

st.set_page_config(page_title="Intelligent Traffic Monitoring", layout="wide")

st.title("Intelligent Traffic Monitoring System")
st.markdown("Detect • Track • Count • Analyze Traffic")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

st.sidebar.header("Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.4, 0.05)
file_type = st.sidebar.radio("Input Type", ["Image", "Video"])

uploaded_file = st.sidebar.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded_file is not None:
    
    if file_type == "Image":
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Run Detection"):
            with st.spinner("Detecting vehicles..."):
                img_array = np.array(image)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                results = model(img_array, conf=confidence, classes=list(VEHICLE_CLASSES.keys()), verbose=False)[0]
                annotated = results.plot()
                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                counts = defaultdict(int)
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in VEHICLE_CLASSES:
                        counts[VEHICLE_CLASSES[cls_id]] += 1

                st.image(annotated, caption="Detection Result", use_container_width=True)

                st.subheader("Detection Summary")
                if counts:
                    for name, count in counts.items():
                        st.write(f"**{name.capitalize()}**: {count}")
                    st.write(f"**Total Vehicles**: {sum(counts.values())}")
                else:
                    st.write("No vehicles detected.")

    elif file_type == "Video":
        # Save uploaded video properly
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()
        video_path = tfile.name

        st.video(uploaded_file)

        if st.button("Run Tracking & Analysis"):
            with st.spinner("Processing video... Please wait"):
                try:
                    unique_vehicles = defaultdict(set)
                    frame_counts = []

                    results = model.track(
                        source=video_path,
                        conf=confidence,
                        classes=list(VEHICLE_CLASSES.keys()),
                        persist=True,
                        stream=True,
                        verbose=False
                    )

                    for r in results:
                        current = 0
                        if r.boxes is not None and r.boxes.id is not None:
                            for track_id, cls_id in zip(
                                r.boxes.id.int().cpu().tolist(),
                                r.boxes.cls.int().cpu().tolist()
                            ):
                                if cls_id in VEHICLE_CLASSES:
                                    unique_vehicles[VEHICLE_CLASSES[cls_id]].add(track_id)
                                    current += 1
                        frame_counts.append(current)

                    total_unique = sum(len(ids) for ids in unique_vehicles.values())
                    avg_on_screen = sum(frame_counts) / len(frame_counts) if frame_counts else 0

                    if avg_on_screen < 3:
                        condition = "Free Flow"
                    elif avg_on_screen < 7:
                        condition = "Moderate"
                    elif avg_on_screen < 12:
                        condition = "Heavy"
                    else:
                        condition = "Congested"

                    st.success("Processing Complete!")
                    st.subheader("Traffic Analysis Results")
                    st.metric("Total Unique Vehicles", total_unique)
                    st.metric("Avg Vehicles on Screen", f"{avg_on_screen:.1f}")
                    st.metric("Traffic Condition", condition)

                    st.write("### Breakdown by Type")
                    for name, ids in unique_vehicles.items():
                        st.write(f"- **{name.capitalize()}**: {len(ids)}")

                except Exception as e:
                    st.error(f"Error processing video: {e}")
                finally:
                    # Clean up temp file
                    if os.path.exists(video_path):
                        os.remove(video_path)

else:
    st.info("Please upload an image or video to begin.")