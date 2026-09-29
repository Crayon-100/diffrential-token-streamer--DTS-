"""DAVIS 2016 480p Dataset Downloader

Downloads and extracts the official DAVIS 2016 (Densely Annotated VIdeo Segmentation)
480p benchmark dataset into the `data/DAVIS` directory.

URL: https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-data.zip
Expected directory structure:
    data/DAVIS/
        ├── Annotations/480p/
        ├── ImageSets/480p/
        └── JPEGImages/480p/

Usage:
    python data/download_davis.py
    # or with uv:
    uv run python data/download_davis.py
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path

DAVIS_2016_URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-data.zip"
TARGET_DIR = Path(__file__).resolve().parent / "DAVIS"
ZIP_FILE = Path(__file__).resolve().parent / "DAVIS-data.zip"


def download_progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\rDownloading DAVIS 2016: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
    else:
        sys.stdout.write(f"\rDownloading DAVIS 2016: {downloaded / (1024 * 1024):.1f} MB")
    sys.stdout.flush()


def main() -> None:
    TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)

    if (TARGET_DIR / "JPEGImages" / "480p").exists():
        print(f"[INFO] DAVIS dataset already exists at {TARGET_DIR}. Skipping download.")
        return

    print(f"[INFO] Target download URL: {DAVIS_2016_URL}")
    print(f"[INFO] Destination: {TARGET_DIR}")

    if not ZIP_FILE.exists():
        print("[INFO] Starting download (~350 MB)...")
        try:
            urllib.request.urlretrieve(DAVIS_2016_URL, ZIP_FILE, reporthook=download_progress)
            print("\n[INFO] Download completed.")
        except Exception as e:
            print(f"\n[ERROR] Failed to download from {DAVIS_2016_URL}: {e}")
            print("[HINT] You can manually download DAVIS 2016 480p from:")
            print("       https://davischallenge.org/davis2016/code.html")
            print(f"       and extract into {TARGET_DIR}")
            sys.exit(1)
    else:
        print(f"[INFO] Found existing zip archive: {ZIP_FILE}")

    print(f"[INFO] Extracting {ZIP_FILE} to {TARGET_DIR.parent}...")
    try:
        with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall(TARGET_DIR.parent)
        print("[INFO] Extraction complete.")
    finally:
        if ZIP_FILE.exists():
            print(f"[INFO] Cleaning up temporary archive: {ZIP_FILE}")
            ZIP_FILE.unlink()

    print(f"[SUCCESS] DAVIS 2016 dataset ready at {TARGET_DIR}")


if __name__ == "__main__":
    main()
