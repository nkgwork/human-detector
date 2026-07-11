"""
Bulk downloader for test videos — downloads ONLY from the two given
playlists, filtered to 5-10 minute duration.

For personal testing purposes only.

Setup:
    pip install yt-dlp

Usage:
    python download_long_test_videos.py
"""

import subprocess
from pathlib import Path

# --- Duration filter: only keep videos between these lengths (in seconds) ---
MIN_DURATION_SEC = 300   # 5 minutes
MAX_DURATION_SEC = 600   # 10 minutes

HUMAN_PLAYLIST = "https://youtube.com/playlist?list=PL9bw4S5ePsEE0jGfUgUMvzeWAaMPcqHL9&si=QnrFrhHDdLaF3ljN"
NON_HUMAN_PLAYLIST = "https://youtube.com/playlist?list=PL4Gr5tOAPttKUXrXjulSCYa-L4xIwDyTi&si=UOyfXYDpGgyQg1U9"

VIDEOS_LIMIT = 50  # max videos to pull from each playlist (after duration filtering, could be fewer)

OUTPUT_DIR = Path("test_videos_long")

DURATION_FILTER = f"duration > {MIN_DURATION_SEC} & duration < {MAX_DURATION_SEC}"


def download_playlist(playlist_url, category, limit):
    out_dir = OUTPUT_DIR / category
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{category}] Downloading from playlist (5-10 min videos only)...")
    subprocess.run([
        "yt-dlp",
        "-f", "best[height<=480]",
        "--match-filter", DURATION_FILTER,
        "--yes-playlist",
        "--playlist-end", str(limit),
        "-o", str(out_dir / "%(title).60s.%(ext)s"),
        playlist_url,
    ])


def download_all():
    download_playlist(HUMAN_PLAYLIST, "human", VIDEOS_LIMIT)
    download_playlist(NON_HUMAN_PLAYLIST, "non_human", VIDEOS_LIMIT)

    print(f"\nDone. Videos saved under: {OUTPUT_DIR}/")
    print(f"Only videos between {MIN_DURATION_SEC//60}-{MAX_DURATION_SEC//60} minutes were kept.")


if __name__ == "__main__":
    download_all()