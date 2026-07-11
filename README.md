# Human Object Detection — Video Classifier

Detects whether a person appears in a video and classifies it as **Human**
or **Non-Human**. Built on pretrained **YOLOv8** (COCO dataset), with a web
demo dashboard for quick, visual testing.

## Project Structure

```
human_detector/
├── app.py                       # Web demo dashboard (Streamlit)
├── object_detector.py           # Core detection logic (CLI + importable)
├── requirements.txt             # Python dependencies
├── yolov8n.pt                   # YOLOv8 model weights (auto-downloaded)
├── test_videos_short/           # Sample short test clips
├── test_videos_long/            # Sample 5-10 min test videos
└── venv/                        # Python virtual environment
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

First run auto-downloads YOLOv8 weights (~6MB) — needs internet once.

## Usage

### Option 1 — Web Demo Dashboard (recommended for demos)

```bash
streamlit run app.py
```

Opens a browser UI at `http://localhost:8501` with:
- **Single Video tab** — upload one video, see the result instantly (label, confidence, video preview)
- **Batch tab** — upload multiple videos, get a results table with summary counts

### Option 2 — Command Line

Test a single video:
```bash
python object_detector.py --single path/to/video.mp4
```

Batch process a folder of videos (auto-segregates into Human / Non-Human):
```bash
python object_detector.py --input input_videos --output output
```

Output structure:
```
output/
├── Human/
├── Non-Human/
├── Errors/
└── classification_report.csv
```

## Tuning

| Flag | Default | Purpose |
|---|---|---|
| `--confidence` | 0.5 | Detection confidence threshold |
| `--sample-rate` | 10 | Checks every Nth frame — lower is more thorough but slower |
| `--model` | yolov8n.pt | Swap to `yolov8s.pt` / `yolov8m.pt` for higher accuracy |
| `--move` | off | Move files instead of copying |

## How It Works

Frames are sampled at the given rate and run through YOLOv8's detector,
restricted to the "person" class. The moment a person is found above the
confidence threshold, the video is labeled **Human** and scanning stops
early. If no person is found after sampling, it's labeled **Non-Human**.