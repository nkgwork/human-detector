"""
Diagnostic script — prints per-frame timing to figure out WHERE the
slowdown is happening (model loading, video decoding, or inference itself).

Usage:
    python diagnose_speed.py --video path/to/video.mp4
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def diagnose(video_path: Path, confidence: float = 0.45):
    print(f"Video: {video_path}")

    # --- Check basic video properties ---
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration (calculated): {total_frames/fps if fps else 'unknown'}s")
    print()

    # --- Time model loading ---
    t0 = time.time()
    model = YOLO("yolov8n.pt")
    print(f"Model load time: {round(time.time() - t0, 2)}s")
    print()

    # --- Time tracking, frame by frame, with live progress ---
    print("Starting frame-by-frame tracking (printing every 10 frames)...")
    t0 = time.time()
    frame_count = 0

    results = model.track(
        source=str(video_path),
        classes=[0],
        conf=confidence,
        persist=True,
        stream=True,
        verbose=False,
        tracker="bytetrack.yaml",
    )

    for r in results:
        frame_count += 1
        if frame_count % 10 == 0:
            elapsed = time.time() - t0
            fps_processing = frame_count / elapsed
            print(f"  Frame {frame_count}/{total_frames} | "
                  f"Elapsed: {round(elapsed, 1)}s | "
                  f"Processing speed: {round(fps_processing, 2)} frames/sec")

    total_elapsed = round(time.time() - t0, 2)
    print()
    print(f"DONE. Total frames processed: {frame_count}")
    print(f"Total tracking time: {total_elapsed}s")
    print(f"Average speed: {round(frame_count/total_elapsed, 2)} frames/sec")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--confidence", type=float, default=0.45)
    args = parser.parse_args()
    diagnose(Path(args.video), args.confidence)
    