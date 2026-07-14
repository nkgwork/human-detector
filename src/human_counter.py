"""
Accurate Human Counter — full-video scan using tracking, to count unique
individuals appearing in a video (not just detect presence).

Core business logic used by BOTH the Single Video and Batch tabs in app.py.

Key behavior:
    - Scans EVERY frame (no early exit, no frame skipping) — required for
      tracking continuity. Skipping frames can cause the tracker to lose
      a person and assign them a new ID when they reappear, inflating the
      count.
    - Uses YOLOv8's built-in ByteTrack tracker: each detected person gets
      a persistent ID as long as they stay continuously visible/trackable.
    - Final unique count = number of distinct track IDs seen across the
      whole video.

Example: if a person is detected in 5 separate frames, but it's the same
person reappearing (2 of those detections are the same individual), the
unique count correctly reflects 4 distinct people — not 5 raw detections.

LIMITATION (be upfront about this in the demo):
    This is track-based, not true re-identification. If a person leaves
    the frame entirely and comes back much later, they may be counted as
    a new person, since the tracker has no memory of their appearance
    after they disappear. True Re-ID (recognizing "this is the same
    person from earlier") would need an additional embedding-matching
    model layered on top — a separate, bigger feature.

Setup:
    pip install ultralytics opencv-python-headless lap
"""

import time
from pathlib import Path

from ultralytics import YOLO

COCO_PERSON_CLASS_ID = 0


class AccurateHumanCounter:
    """Wraps YOLOv8 tracking to count unique humans across an entire video."""

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.45):
        self.model = YOLO(model_name)
        self.confidence = confidence

    def analyze_video(self, video_path: Path) -> dict:
        """
        Full-video scan. Returns label (Human/Non-Human), unique person
        count, total raw detections, frames processed, and time taken.
        """
        t0 = time.time()

        seen_track_ids = set()
        raw_detection_count = 0
        frame_count = 0

        try:
            results = self.model.track(
                source=str(video_path),
                classes=[COCO_PERSON_CLASS_ID],
                conf=self.confidence,
                persist=True,
                stream=True,       # memory-efficient for long videos
                verbose=False,
                tracker="bytetrack.yaml",  # lighter tracker, avoids slow ORB motion-matching
                                            # that BoT-SORT (default) does — fixes
                                            # "not enough matching points" slowdowns/hangs
            )

            for r in results:
                frame_count += 1
                if r.boxes is not None and len(r.boxes) > 0:
                    raw_detection_count += len(r.boxes)
                    if r.boxes.id is not None:
                        ids = r.boxes.id.int().tolist()
                        seen_track_ids.update(ids)

        except Exception as e:
            return {"status": "error", "reason": str(e)}

        elapsed = round(time.time() - t0, 2)
        unique_count = len(seen_track_ids)

        return {
            "status": "ok",
            "label": "Human" if unique_count > 0 else "Non-Human",
            "unique_human_count": unique_count,
            "raw_detections": raw_detection_count,
            "frames_processed": frame_count,
            "elapsed_sec": elapsed,
        }