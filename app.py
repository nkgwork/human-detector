"""
Demo Dashboard — Human Object Detection in Videos
====================================================
A simple, professional web UI for demoing the video classifier.
Runs locally, no deployment needed.

Setup:
    pip install streamlit
    (object_detector.py must be in the same folder)

Run:
    streamlit run app.py

Opens automatically in your browser at http://localhost:8501
"""

import tempfile
from pathlib import Path

import streamlit as st

from src.object_detector import HumanDetector

st.set_page_config(
    page_title="Human Object Detection - Video Classifier",
    page_icon="🎥",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🎥 Human Object Detection in Videos")
st.caption("Upload a video to check whether it contains a person. Powered by YOLOv8.")

st.divider()

# ---------------------------------------------------------------------------
# Sidebar settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    confidence = st.slider("Detection confidence threshold", 0.1, 0.9, 0.5, 0.05)
    sample_rate = st.slider("Frame sample rate (check every Nth frame)", 1, 30, 10)
    st.caption("Lower sample rate = more thorough scan, but slower.")
    st.divider()
    st.caption("Model: YOLOv8n (nano) — pretrained on COCO dataset")


@st.cache_resource
def load_detector(conf):
    return HumanDetector(model_name="yolov8n.pt", confidence=conf)


# ---------------------------------------------------------------------------
# Single video demo
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📹 Single Video Demo", "📁 Batch (Multiple Videos)"])

with tab1:
    uploaded_file = st.file_uploader(
        "Upload a video file", type=["mp4", "avi", "mov", "mkv", "webm", "m4v"], key="single"
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.video(uploaded_file)

        with col2:
            with st.spinner("Analyzing video..."):
                # Save uploaded file to a temp path so OpenCV can read it
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                detector = load_detector(confidence)
                result = detector.classify_video(Path(tmp_path), sample_rate=sample_rate)

            if result["status"] == "ok":
                label = result["label"]
                if label == "Human":
                    st.success(f"### ✅ {label} Detected")
                else:
                    st.info(f"### ⬜ {label}")

                m1, m2, m3 = st.columns(3)
                m1.metric("Confidence", f"{result['confidence']*100:.1f}%")
                m2.metric("Frames Checked", result["frames_checked"])
                m3.metric("Total Frames", result["total_frames"])

                if result["found_at_frame"] is not None:
                    st.caption(f"Person first detected at frame {result['found_at_frame']} ({result['fps']} fps)")
            else:
                st.error(f"Could not process video: {result.get('reason', 'unknown error')}")

# ---------------------------------------------------------------------------
# Batch demo
# ---------------------------------------------------------------------------
with tab2:
    uploaded_files = st.file_uploader(
        "Upload multiple videos",
        type=["mp4", "avi", "mov", "mkv", "webm", "m4v"],
        accept_multiple_files=True,
        key="batch",
    )

    if uploaded_files:
        if st.button(f"Classify {len(uploaded_files)} videos", type="primary"):
            detector = load_detector(confidence)
            results = []
            progress = st.progress(0, text="Starting...")

            for i, f in enumerate(uploaded_files):
                progress.progress((i) / len(uploaded_files), text=f"Processing {f.name}...")

                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix) as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name

                result = detector.classify_video(Path(tmp_path), sample_rate=sample_rate)
                results.append({
                    "File": f.name,
                    "Label": result.get("label", "ERROR"),
                    "Confidence": f"{result.get('confidence', 0)*100:.1f}%" if result["status"] == "ok" else "-",
                    "Frames Checked": result.get("frames_checked", "-"),
                })

            progress.progress(1.0, text="Done!")

            human_count = sum(1 for r in results if r["Label"] == "Human")
            non_human_count = sum(1 for r in results if r["Label"] == "Non-Human")

            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Videos", len(results))
            c2.metric("Human", human_count)
            c3.metric("Non-Human", non_human_count)

            st.dataframe(results, use_container_width=True, hide_index=True)

st.divider()
st.caption("Demo dashboard for the human object detection classifier — for evaluation purposes.")
