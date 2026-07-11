#!/usr/bin/env python3
"""
Human Video Classifier
=======================
Scans videos in an input folder, detects whether a human/person appears
in each video using a pretrained YOLOv8 model, and segregates the videos
into 'Human' and 'Non-Human' output folders (copy, original file untouched).

Usage:
    python classifier.py --input input_videos --output output
    python classifier.py --input input_videos --output output --confidence 0.5 --sample-rate 10
    python classifier.py --input input_videos --output output --move   # move instead of copy
    python classifier.py --single path/to/video.mp4                    # test a single file, no copy

Author: built for NKG's human/non-human video classification task
"""

import argparse
import csv
import logging
import shutil
import sys
import time
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("human_classifier")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
COCO_PERSON_CLASS_ID = 0  # "person" is class 0 in the standard COCO dataset


class HumanDetector:
    """Wraps a YOLOv8 model to answer: does this video contain a person?"""

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.5):
        from ultralytics import YOLO  # imported lazily so --help works without torch installed

        log.info(f"Loading model '{model_name}' (first run downloads weights, ~6MB)...")
        self.model = YOLO(model_name)
        self.confidence = confidence
        log.info("Model loaded.")

    def frame_has_person(self, frame) -> tuple[bool, float]:
        """Returns (person_found, max_confidence) for a single frame."""
        results = self.model(
            frame,
            classes=[COCO_PERSON_CLASS_ID],
            conf=self.confidence,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            max_conf = float(boxes.conf.max())
            return True, max_conf
        return False, 0.0

    def classify_video(self, video_path: Path, sample_rate: int = 10, max_frames_checked: int = 300):
        """
        Samples frames from the video (every `sample_rate`-th frame, capped at
        `max_frames_checked` samples) and returns a result dict as soon as a
        person is found, or after exhausting the sample budget.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"status": "error", "reason": "could not open video file"}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0

        frame_idx = 0
        checked = 0
        best_conf = 0.0
        person_found = False
        found_at_frame = None

        while checked < max_frames_checked:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            found, conf = self.frame_has_person(frame)
            checked += 1
            if found:
                person_found = True
                best_conf = conf
                found_at_frame = frame_idx
                break  # early exit — no need to scan the rest

            frame_idx += sample_rate
            if total_frames and frame_idx >= total_frames:
                break

        cap.release()

        return {
            "status": "ok",
            "label": "Human" if person_found else "Non-Human",
            "confidence": round(best_conf, 3),
            "frames_checked": checked,
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "found_at_frame": found_at_frame,
        }


def discover_videos(input_dir: Path):
    return sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def place_file(src: Path, dest_dir: Path, move: bool):
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if move:
        shutil.move(str(src), str(dest))
    else:
        shutil.copy2(str(src), str(dest))
    return dest


def run_batch(input_dir: Path, output_dir: Path, confidence: float, sample_rate: int, move: bool, model_name: str):
    videos = discover_videos(input_dir)
    if not videos:
        log.warning(f"No video files found in '{input_dir}'. Supported: {sorted(VIDEO_EXTENSIONS)}")
        return

    log.info(f"Found {len(videos)} video(s) to classify.")
    detector = HumanDetector(model_name=model_name, confidence=confidence)

    human_dir = output_dir / "Human"
    non_human_dir = output_dir / "Non-Human"
    error_dir = output_dir / "Errors"

    results = []
    start = time.time()

    for i, video_path in enumerate(videos, 1):
        log.info(f"[{i}/{len(videos)}] Processing: {video_path.name}")
        t0 = time.time()
        result = detector.classify_video(video_path, sample_rate=sample_rate)
        elapsed = round(time.time() - t0, 2)

        if result["status"] == "error":
            log.error(f"  -> ERROR: {result['reason']}")
            place_file(video_path, error_dir, move)
            results.append({"file": video_path.name, "label": "ERROR", "confidence": "",
                             "frames_checked": "", "elapsed_sec": elapsed})
            continue

        label = result["label"]
        dest_dir = human_dir if label == "Human" else non_human_dir
        place_file(video_path, dest_dir, move)

        log.info(
            f"  -> {label} (confidence={result['confidence']}, "
            f"checked {result['frames_checked']} frames in {elapsed}s)"
        )
        results.append({
            "file": video_path.name,
            "label": label,
            "confidence": result["confidence"],
            "frames_checked": result["frames_checked"],
            "elapsed_sec": elapsed,
        })

    total_elapsed = round(time.time() - start, 2)
    write_report(output_dir, results)

    human_count = sum(1 for r in results if r["label"] == "Human")
    non_human_count = sum(1 for r in results if r["label"] == "Non-Human")
    error_count = sum(1 for r in results if r["label"] == "ERROR")

    log.info("=" * 60)
    log.info(f"DONE in {total_elapsed}s | Human: {human_count} | Non-Human: {non_human_count} | Errors: {error_count}")
    log.info(f"Report saved to: {output_dir / 'classification_report.csv'}")


def write_report(output_dir: Path, results: list):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "classification_report.csv"
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "label", "confidence", "frames_checked", "elapsed_sec"])
        writer.writeheader()
        writer.writerows(results)


def run_single(video_path: Path, confidence: float, sample_rate: int, model_name: str):
    detector = HumanDetector(model_name=model_name, confidence=confidence)
    result = detector.classify_video(video_path, sample_rate=sample_rate)
    log.info(f"Result for '{video_path.name}': {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Classify videos as Human / Non-Human and segregate them.")
    parser.add_argument("--input", type=str, default="input_videos", help="Folder containing videos to classify")
    parser.add_argument("--output", type=str, default="output", help="Folder to place Human/Non-Human results")
    parser.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold (0-1)")
    parser.add_argument("--sample-rate", type=int, default=10, help="Check every Nth frame (higher = faster, less thorough)")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model weights (yolov8n/s/m/l/x.pt)")
    parser.add_argument("--single", type=str, default=None, help="Test a single video file (no copy/move, just prints result)")
    args = parser.parse_args()

    if args.single:
        run_single(Path(args.single), args.confidence, args.sample_rate, args.model)
    else:
        run_batch(
            input_dir=Path(args.input),
            output_dir=Path(args.output),
            confidence=args.confidence,
            sample_rate=args.sample_rate,
            move=args.move,
            model_name=args.model,
        )


if __name__ == "__main__":
    main()
