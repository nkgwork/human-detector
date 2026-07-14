# """
# Demo Dashboard — Human Object Detection & Counting in Videos
# ================================================================
# Two tabs, both powered by the same accurate, full-video-scan counting logic:
#     1. Single Video — upload one video, see label + unique human count
#     2. Batch — upload multiple videos, see a results table with counts

# Setup:
#     pip install streamlit
#     (src/human_counter.py must be present alongside this file)

# Run:
#     streamlit run app.py
# """

# import tempfile
# from pathlib import Path

# import streamlit as st

# from src.human_counter import AccurateHumanCounter

# st.set_page_config(
#     page_title="Human Object Detection - Video Classifier",
#     page_icon="🎥",
#     layout="wide",
# )

# st.title("🎥 Human Object Detection & Counting")
# st.caption("Full-video scan — detects and counts unique people, avoiding double-counting repeats.")

# st.divider()

# with st.sidebar:
#     st.header("Settings")
#     confidence = st.slider("Detection confidence threshold", 0.1, 0.9, 0.45, 0.05)
#     st.caption("Lower = more sensitive (more false positives). Higher = stricter (may miss less-visible people).")
#     st.divider()
#     st.caption("This scans the ENTIRE video, frame by frame, for accuracy. Longer videos will take longer to process.")
#     st.caption("Model: YOLOv8n + ByteTrack (unique person tracking)")


# @st.cache_resource
# def load_counter(conf):
#     return AccurateHumanCounter(model_name="yolov8n.pt", confidence=conf)


# def render_result(result):
#     if result["status"] != "ok":
#         st.error(f"Could not process video: {result.get('reason', 'unknown error')}")
#         return

#     label = result["label"]
#     if label == "Human":
#         st.success(f"### {label} — {result['unique_human_count']} unique person(s) detected")
#     else:
#         st.info(f"### {label} — no person detected")

#     m1, m2, m3 = st.columns(3)
#     m1.metric("Unique Humans", result["unique_human_count"])
#     m2.metric("Raw Detections", result["raw_detections"])
#     m3.metric("Processing Time", f"{result['elapsed_sec']}s")

#     if result["raw_detections"] > result["unique_human_count"]:
#         repeats = result["raw_detections"] - result["unique_human_count"]
#         st.caption(f"{repeats} repeat detection(s) of already-seen individuals were not double-counted.")


# tab1, tab2 = st.tabs(["Single Video", "Batch (Multiple Videos)"])

# # ---------------------------------------------------------------------------
# # Tab 1: Single Video
# # ---------------------------------------------------------------------------
# with tab1:
#     uploaded_file = st.file_uploader(
#         "Upload a video file", type=["mp4", "avi", "mov", "mkv", "webm", "m4v"], key="single"
#     )

#     if uploaded_file is not None:
#         col1, col2 = st.columns([1, 1])

#         with col1:
#             st.video(uploaded_file)

#         with col2:
#             with st.spinner("Scanning entire video for accurate count... this may take a while for longer videos."):
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
#                     tmp.write(uploaded_file.read())
#                     tmp_path = tmp.name

#                 counter = load_counter(confidence)
#                 result = counter.analyze_video(Path(tmp_path))

#             render_result(result)

# # ---------------------------------------------------------------------------
# # Tab 2: Batch
# # ---------------------------------------------------------------------------
# with tab2:
#     uploaded_files = st.file_uploader(
#         "Upload multiple videos",
#         type=["mp4", "avi", "mov", "mkv", "webm", "m4v"],
#         accept_multiple_files=True,
#         key="batch",
#     )

#     if uploaded_files:
#         if st.button(f"Analyze {len(uploaded_files)} videos", type="primary"):
#             counter = load_counter(confidence)
#             results = []
#             progress = st.progress(0, text="Starting...")

#             for i, f in enumerate(uploaded_files):
#                 progress.progress(i / len(uploaded_files), text=f"Scanning {f.name} (full video)...")

#                 with tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix) as tmp:
#                     tmp.write(f.read())
#                     tmp_path = tmp.name

#                 result = counter.analyze_video(Path(tmp_path))
#                 results.append({
#                     "File": f.name,
#                     "Label": result.get("label", "ERROR"),
#                     "Unique Humans": result.get("unique_human_count", "-"),
#                     "Raw Detections": result.get("raw_detections", "-"),
#                     "Time (s)": result.get("elapsed_sec", "-"),
#                 })

#             progress.progress(1.0, text="Done!")

#             human_videos = sum(1 for r in results if r["Label"] == "Human")
#             total_unique_people = sum(r["Unique Humans"] for r in results if isinstance(r["Unique Humans"], int))

#             st.divider()
#             c1, c2, c3 = st.columns(3)
#             c1.metric("Total Videos", len(results))
#             c2.metric("Videos with Humans", human_videos)
#             c3.metric("Total Unique People (across all videos)", total_unique_people)

#             st.dataframe(results, use_container_width=True, hide_index=True)

# st.divider()
# st.caption("Accurate mode: scans every frame of every video for reliable unique-person counts.")




"""
Demo Dashboard — Human Object Detection & Counting in Videos
================================================================
Two tabs, both powered by the same accurate, full-video-scan counting logic:
    1. Single Video — upload one video, see label + unique human count
    2. Batch — upload multiple videos, see a results table with counts

Setup:
    pip install streamlit
    (src/human_counter.py must be present alongside this file)

Run:
    streamlit run app.py
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

st.title("🎥 Human Object Detection & Counting")
st.caption("Full-video scan — detects and counts unique people, avoiding double-counting repeats.")

st.divider()

with st.sidebar:
    st.header("Settings")
    confidence = st.slider("Detection confidence threshold", 0.1, 0.9, 0.45, 0.05)
    st.caption("Lower = more sensitive (more false positives). Higher = stricter (may miss less-visible people).")
    st.divider()
    st.caption("This scans the ENTIRE video, frame by frame, for accuracy. Longer videos will take longer to process.")
    st.caption("Model: YOLOv8n + ByteTrack (unique person tracking)")


@st.cache_resource
def load_counter(conf):
    return HumanDetector(model_name="yolov8n.pt", confidence=conf)


def render_result(result):
    if result["status"] != "ok":
        st.error(f"Could not process video: {result.get('reason', 'unknown error')}")
        return

    label = result["label"]
    if label == "Human":
        st.success(f"### {label} — {result['unique_human_count']} unique person(s) detected")
    else:
        st.info(f"### {label} — no person detected")

    m1, m2, m3 = st.columns(3)
    m1.metric("Unique Humans", result["unique_human_count"])
    m2.metric("Raw Detections", result["raw_detections"])
    m3.metric("Processing Time", f"{result['elapsed_sec']}s")

    if result["raw_detections"] > result["unique_human_count"]:
        repeats = result["raw_detections"] - result["unique_human_count"]
        st.caption(f"{repeats} repeat detection(s) of already-seen individuals were not double-counted.")


tab1, tab2 = st.tabs(["Single Video", "Batch (Multiple Videos)"])

# ---------------------------------------------------------------------------
# Tab 1: Single Video
# ---------------------------------------------------------------------------
with tab1:
    uploaded_file = st.file_uploader(
        "Upload a video file", type=["mp4", "avi", "mov", "mkv", "webm", "m4v"], key="single"
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.video(uploaded_file)

        with col2:
            with st.spinner("Scanning entire video for accurate count... this may take a while for longer videos."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                counter = load_counter(confidence)
                result = counter.count_unique_humans(Path(tmp_path))

            render_result(result)

# ---------------------------------------------------------------------------
# Tab 2: Batch
# ---------------------------------------------------------------------------
with tab2:
    uploaded_files = st.file_uploader(
        "Upload multiple videos",
        type=["mp4", "avi", "mov", "mkv", "webm", "m4v"],
        accept_multiple_files=True,
        key="batch",
    )

    if uploaded_files:
        if st.button(f"Analyze {len(uploaded_files)} videos", type="primary"):
            counter = load_counter(confidence)
            results = []
            progress = st.progress(0, text="Starting...")

            for i, f in enumerate(uploaded_files):
                progress.progress(i / len(uploaded_files), text=f"Scanning {f.name} (full video)...")

                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix) as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name

                result = counter.count_unique_humans(Path(tmp_path))
                results.append({
                    "File": f.name,
                    "Label": result.get("label", "ERROR"),
                    "Unique Humans": result.get("unique_human_count", "-"),
                    "Raw Detections": result.get("raw_detections", "-"),
                    "Time (s)": result.get("elapsed_sec", "-"),
                })

            progress.progress(1.0, text="Done!")

            human_videos = sum(1 for r in results if r["Label"] == "Human")
            total_unique_people = sum(r["Unique Humans"] for r in results if isinstance(r["Unique Humans"], int))

            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Videos", len(results))
            c2.metric("Videos with Humans", human_videos)
            c3.metric("Total Unique People (across all videos)", total_unique_people)

            st.dataframe(results, use_container_width=True, hide_index=True)

st.divider()
st.caption("Accurate mode: scans every frame of every video for reliable unique-person counts.")