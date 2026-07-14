"""
Bulk video downloader — accepts EITHER a playlist link OR a single video
link. Automatically detects which type each URL is:
    - Playlist link (contains "list=" in the URL) -> downloads up to
      COUNT videos from that playlist
    - Single video link (no "list=") -> downloads just that one video,
      COUNT is ignored for it

No duration filtering — downloads whatever videos are found, regardless
of length.

For personal testing purposes only.

Setup:
    pip install yt-dlp

Usage:
    Edit the SOURCES list below, then run:
    python download_videos.py
"""

import subprocess
from pathlib import Path

# Add any mix of playlist links and single video links here.
# "count" only matters for playlists (how many videos to pull from it).
# For single video links, "count" is ignored — only that 1 video downloads.
SOURCES = [
    # {
    #     "url": "https://youtube.com/playlist?list=PL9bw4S5ePsEE0jGfUgUMvzeWAaMPcqHL9&si=QnrFrhHDdLaF3ljN",
    #     "category": "human",
    #     "count": 50,
    # },
    # {
    #     "url": "https://youtube.com/playlist?list=PL4Gr5tOAPttKUXrXjulSCYa-L4xIwDyTi&si=UOyfXYDpGgyQg1U9",
    #     "category": "non_human",
    #     "count": 50,
    # },
    # Example of a single video link (count is ignored for these):
    {
        "url": "https://youtu.be/JISqoVBJFcs?si=uMlJA2uJb2t-xxDo",
        "category": "human",
        "count": 1,
    },
]

OUTPUT_DIR = Path("test_videos_long")


def is_playlist_url(url: str) -> bool:
    """Playlist links contain a 'list=' query parameter. Single video
    links (watch?v=... or youtu.be/...) don't."""
    return "list=" in url


def download_source(url: str, category: str, count: int):
    out_dir = OUTPUT_DIR / category
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "yt-dlp",
        "-f", "best[height<=480]",
        "-o", str(out_dir / "%(title).60s.%(ext)s"),
    ]

    if is_playlist_url(url):
        print(f"\n[{category}] Detected PLAYLIST link. Downloading up to {count} videos...")
        cmd += ["--yes-playlist", "--playlist-end", str(count)]
    else:
        print(f"\n[{category}] Detected SINGLE VIDEO link. Downloading 1 video...")
        cmd += ["--no-playlist"]

    cmd.append(url)
    subprocess.run(cmd)


def download_all():
    for source in SOURCES:
        download_source(source["url"], source["category"], source["count"])

    print(f"\nDone. Videos saved under: {OUTPUT_DIR}/")


if __name__ == "__main__":
    download_all()