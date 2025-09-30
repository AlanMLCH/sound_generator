# src/data/download.py
"""
Dataset downloader for MIDI sources.

Configure sources in: data/sources.yaml
Then run: make ingest  (or: python -m src.data.download)

Supports:
- HTTP(S) downloads (zip / tar.gz / tar)
- Optional Kaggle datasets (requires KAGGLE_USERNAME/KAGGLE_KEY and kaggle CLI)

Downloaded archives -> data/external/_downloads/
Extracted contents  -> data/external/<name>/
All *.mid/*.midi    -> data/raw/<name>/ for the pipeline.
"""
import os
import glob
import tarfile
import zipfile
import shutil
import subprocess
from typing import Dict, Any, List

import requests
from tqdm import tqdm
import yaml

from src.common.constants import RAW_DIR

BASE_DATA_DIR = "data"
EXTERNAL_DIR = os.path.join(BASE_DATA_DIR, "external")
DOWNLOADS_DIR = os.path.join(EXTERNAL_DIR, "_downloads")
SOURCES_YAML = os.path.join(BASE_DATA_DIR, "sources.yaml")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(EXTERNAL_DIR, exist_ok=True)
os.makedirs(DOWNLOADS_DIR, exist_ok=True)


def _load_sources() -> List[Dict[str, Any]]:
    if not os.path.exists(SOURCES_YAML):
        print(f"[download] Missing {SOURCES_YAML}. Create it (see sample in repo) and re-run.")
        return []
    with open(SOURCES_YAML, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    sources = cfg.get("sources", [])
    if not sources:
        print("[download] No sources configured in sources.yaml.")
    return sources


def _stream_download(url: str, dest_path: str):
    print(f"[download] Fetching: {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=os.path.basename(dest_path)
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def _extract(archive_path: str, out_dir: str, archive_type: str):
    print(f"[extract] {archive_path} -> {out_dir} ({archive_type})")
    os.makedirs(out_dir, exist_ok=True)
    if archive_type == "zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(out_dir)
    elif archive_type in ("tar", "tar.gz", "tgz"):
        mode = "r:gz" if archive_type in ("tar.gz", "tgz") else "r:"
        with tarfile.open(archive_path, mode) as tf:
            tf.extractall(out_dir)
    else:
        raise ValueError(f"Unsupported archive_type: {archive_type}")


def _copy_midis(src_dir: str, dst_dir: str, patterns: List[str]):
    os.makedirs(dst_dir, exist_ok=True)
    n = 0
    for pat in patterns:
        for fp in glob.glob(os.path.join(src_dir, pat), recursive=True):
            base = os.path.basename(fp)
            out = os.path.join(dst_dir, base)
            if os.path.exists(out):
                stem, ext = os.path.splitext(base)
                k = 1
                while os.path.exists(os.path.join(dst_dir, f"{stem}_{k}{ext}")):
                    k += 1
                out = os.path.join(dst_dir, f"{stem}_{k}{ext}")
            try:
                shutil.copy2(fp, out)
                n += 1
            except Exception as e:
                print(f"[copy] Skip {fp}: {e}")
    print(f"[copy] Copied {n} MIDI files to {dst_dir}")


def _download_http(src: Dict[str, Any]):
    name = src["name"]
    url = src["url"]
    archive_type = src.get("archive_type", "zip")
    subdir = src.get("target_subdir", name)

    ext = {"zip": "zip", "tar": "tar", "tar.gz": "tar.gz", "tgz": "tgz"}.get(archive_type, archive_type)
    dl_name = f"{name}.{ext.replace('.', '')}"
    dl_path = os.path.join(DOWNLOADS_DIR, dl_name)

    if not os.path.exists(dl_path):
        _stream_download(url, dl_path)
    else:
        print(f"[download] Using cached: {dl_path}")

    extract_dir = os.path.join(EXTERNAL_DIR, subdir)
    patterns = src.get("include_glob", ["**/*.mid", "**/*.midi"])
    _extract(dl_path, extract_dir, archive_type, include_glob=patterns)


    raw_target = os.path.join(RAW_DIR, name)
    patterns = src.get("include_glob", ["**/*.mid", "**/*.midi"])
    _copy_midis(extract_dir, raw_target, patterns)


def _download_kaggle(src: Dict[str, Any]):
    """
    Requires Kaggle CLI installed and KAGGLE_USERNAME/KAGGLE_KEY env vars.
    """
    name = src["name"]
    dataset = src["kaggle_dataset"]
    file = src.get("file", "")
    subdir = src.get("target_subdir", name)

    print(f"[kaggle] Downloading {dataset} {'('+file+')' if file else ''}")
    out_dir = os.path.join(EXTERNAL_DIR, subdir)
    os.makedirs(out_dir, exist_ok=True)

    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", out_dir]
    if file:
        cmd += ["-f", file]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("[kaggle] Kaggle CLI not found. Install it or use HTTP sources instead.")
        return
    except subprocess.CalledProcessError as e:
        print(f"[kaggle] Download failed: {e}")
        return

    # Unzip any zips
    for z in glob.glob(os.path.join(out_dir, "*.zip")):
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(out_dir)
        os.remove(z)

    raw_target = os.path.join(RAW_DIR, name)
    patterns = src.get("include_glob", ["**/*.mid", "**/*.midi"])
    _copy_midis(out_dir, raw_target, patterns)


def run():
    sources = _load_sources()
    if not sources:
        return

    for src in sources:
        try:
            stype = src.get("type", "http")
            if stype == "http":
                _download_http(src)
            elif stype == "kaggle":
                _download_kaggle(src)
            else:
                print(f"[download] Unknown source type '{stype}' for {src.get('name')}")
        except Exception as e:
            print(f"[download] Failed source '{src.get('name','?')}': {e}")

    print(f"[download] Done.\n  RAW: {RAW_DIR}\n  EXTERNAL: {EXTERNAL_DIR}")


if __name__ == "__main__":
    run()
