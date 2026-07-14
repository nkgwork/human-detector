# Human Object Detection & Counting — Video Analysis Tool

Detects humans in a video, classifies it as **Human** / **Non-Human**, and
counts **unique individuals** appearing in it — avoiding double-counting
when the same person reappears (after occlusion, leaving frame, etc.).

Built on **YOLOv8**, with tracking + appearance-based Re-Identification for
accurate unique counting.

---

## Project Structure

```
human_detector/
├── .streamlit/
│   └── config.toml              # upload size limit (currently 4GB), UI config
├── .gitignore
├── README.md
├── requirements.txt
├── app.py                       # Web demo dashboard (Streamlit) — entry point
├── src/
│   ├── __init__.py
│   ├── object_detector.py       # Core detection + classification + counting logic
│   ├── custom_bytetrack.yaml    # Tracker config (motion-based, fast)
│   └── custom_botsort_reid.yaml # Tracker config (appearance-based Re-ID, more accurate)
├── scripts/
│   └── download_videos.py       # Test video downloader (playlist or single video, auto-detected)
├── data/
│   ├── input_videos/             # Drop real videos here for batch processing
│   ├── test_videos_short/
│   └── test_videos_long/
├── output/                       # Batch classification results (gitignored)
├── docs/
│   └── demo_script.md            # Talking points / walkthrough script for demos
└── tests/                        # (reserved for future unit tests)
```

---

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

First run auto-downloads YOLOv8 weights (~6MB) and Re-ID model weights —
needs internet once.

---

## Usage

### Web Demo Dashboard
```bash
streamlit run app.py
```
Two tabs — Single Video and Batch — both use the same accurate,
full-video-scan counting logic under the hood.

### CLI — Accurate unique human count (full video scan, slower)
```bash
python -m src.object_detector --count path/to/video.mp4
```

### CLI — Batch classify a folder
```bash
python -m src.object_detector --input data/input_videos --output output
```

---

## Versioning & Branching

- `main` — stable, demo-ready code
- `dev` — active development
- `feature/*` — individual features, merged into `dev` via PR, then `dev` → `main`

---

## Configuration

- **Upload size limit:** `.streamlit/config.toml` → `maxUploadSize` (in MB)
- **Detection confidence / tracker choice:** adjustable via CLI flags (`--confidence`) or by editing the tracker config referenced in `src/object_detector.py`