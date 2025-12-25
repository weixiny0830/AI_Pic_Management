# engine.py (Lazy-Init / Fast UI Startup)
# ============================================================
# GPU-Accelerated Local Photo Manager (CLIP-Based)
#
# Key upgrades vs. eager-init version:
# - LAZY LOAD CLIP + CUDA: UI opens fast; model loads only when needed (first image scoring)
# - Robust Windows path normalization (prevents mixed slash issues)
# - Soft stop (StopFlag)
# - Progress + log callbacks (for UI)
# - Existence checks before processing each file
# - CSV audit log
#
# Notes:
# - First time you click "Start", CLIP will load (can take a few seconds).
# - Subsequent runs in the same process are faster.
# ============================================================

import os
import csv
import shutil
from datetime import datetime
from typing import Callable, Optional, Dict, Set, Tuple

import torch
import clip
from PIL import Image, ExifTags
from send2trash import send2trash
import pillow_heif

# ----------------------------
# Defaults / configuration
# ----------------------------
DEFAULT_IMAGE_EXT: Set[str] = {".jpg", ".jpeg", ".png", ".heic", ".gif"}
DEFAULT_VIDEO_EXT: Set[str] = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".mts", ".m2ts"}
DEFAULT_LOG_PREFIX = "_ai_organizer_log_"

# CLIP prompts: first 5 = screenshot-related
TEXTS = [
    "a smartphone screenshot",
    "a chat app interface",
    "a webpage screenshot",
    "a document screenshot",
    "a computer screen interface",
    "a real-life photograph",
    "a family photo",
    "a portrait photo",
    "a landscape photo",
]

EXIF_TAGS = ExifTags.TAGS

# Enable HEIC support
pillow_heif.register_heif_opener()

# ----------------------------
# Types
# ----------------------------
LogCB = Optional[Callable[[str], None]]
ProgressCB = Optional[Callable[[int, int, str], None]]  # done, total, stage


# ----------------------------
# Windows path normalization
# ----------------------------
def norm_win_path(p: str) -> str:
    """Normalize to absolute Windows-style path (prevents mixed slash bugs)."""
    return os.path.normpath(os.path.abspath(p))


def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------
# Lazy CLIP initialization
# ----------------------------
_DEVICE: Optional[str] = None
_MODEL = None
_PREPROCESS = None
_TEXT_TOKENS = None


def is_cuda_available() -> bool:
    # Safe to call without initializing CLIP model.
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def get_clip_bundle() -> Tuple[str, object, object, object]:
    """
    Lazily initialize CLIP model & tokens.
    Returns (device, model, preprocess, text_tokens).
    """
    global _DEVICE, _MODEL, _PREPROCESS, _TEXT_TOKENS
    if _MODEL is None:
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL, _PREPROCESS = clip.load("ViT-B/32", device=_DEVICE)
        _TEXT_TOKENS = clip.tokenize(TEXTS).to(_DEVICE)
    return _DEVICE, _MODEL, _PREPROCESS, _TEXT_TOKENS


def get_device_str() -> str:
    """
    Returns best-available device string without forcing CLIP to load.
    Useful for UI header. (Final runtime device logged when model loads.)
    """
    return "cuda" if is_cuda_available() else "cpu"


# ----------------------------
# Media date + file ops
# ----------------------------
def get_media_date(path: str, image_ext: Set[str]) -> datetime:
    """
    Try EXIF DateTimeOriginal for images. Fallback to Windows file creation time.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in image_ext:
        try:
            img = Image.open(path)
            exif = img._getexif()
            if exif:
                for tag, val in exif.items():
                    if EXIF_TAGS.get(tag) == "DateTimeOriginal":
                        return datetime.strptime(val, "%Y:%m:%d %H:%M:%S")
        except Exception:
            pass

    return datetime.fromtimestamp(os.path.getctime(path))


@torch.no_grad()
def clip_screenshot_score(image_path: str) -> float:
    """
    Returns screenshot probability score (0–1).
    Higher means more likely to be a screenshot.
    """
    try:
        device, model, preprocess, text_tokens = get_clip_bundle()

        img = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(img).unsqueeze(0).to(device)

        image_feat = model.encode_image(image_tensor)
        text_feat = model.encode_text(text_tokens)

        probs = (image_feat @ text_feat.T).softmax(dim=-1)[0]
        return float(probs[:5].sum().item())
    except Exception:
        # If image is corrupted/unreadable, treat as non-screenshot
        return 0.0


def unique_target_path(target_dir: str, filename: str) -> str:
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(target_dir, filename)
    if not os.path.exists(candidate):
        return candidate

    i = 1
    while True:
        candidate = os.path.join(target_dir, f"{base}_{i}{ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1


def move_file(src: str, dst_dir: str) -> str:
    safe_makedirs(dst_dir)
    dst = unique_target_path(dst_dir, os.path.basename(src))
    shutil.move(src, dst)
    return dst


def is_in_target_folders(path: str, photos_dir: str, videos_dir: str, review_dir: str) -> bool:
    norm = os.path.normpath(path)
    return (
        norm.startswith(os.path.normpath(photos_dir) + os.sep)
        or norm.startswith(os.path.normpath(videos_dir) + os.sep)
        or norm.startswith(os.path.normpath(review_dir) + os.sep)
    )


# ----------------------------
# Soft stop
# ----------------------------
class StopFlag:
    def __init__(self):
        self._stop = False

    def request_stop(self):
        self._stop = True

    def is_stopped(self) -> bool:
        return self._stop


# ----------------------------
# Main entry
# ----------------------------
def run_job(
    root_dir: str,
    screenshot_threshold: float = 0.80,
    review_threshold: float = 0.60,
    dry_run: bool = False,
    skip_target_folders: bool = True,
    image_ext: Optional[Set[str]] = None,
    video_ext: Optional[Set[str]] = None,
    log_prefix: str = DEFAULT_LOG_PREFIX,
    log_cb: LogCB = None,
    progress_cb: ProgressCB = None,
    stop_flag: Optional[StopFlag] = None,
) -> Dict:
    """
    Run the organizer/cleaner.

    - screenshot_threshold: >= this => screenshot => Recycle Bin
    - review_threshold: in [review_threshold, screenshot_threshold) => _AI_REVIEW
    - else => photo => Photos/YYYY/YYYY-MM
    - videos => Videos/YYYY/YYYY-MM
    - other files => Recycle Bin
    - Writes a CSV log under root_dir.

    progress_cb(done, total, stage) provides UI progress.
    log_cb(msg) provides UI log output.
    stop_flag allows soft stop.
    """

    root_dir = norm_win_path(root_dir)

    image_ext = image_ext or DEFAULT_IMAGE_EXT
    video_ext = video_ext or DEFAULT_VIDEO_EXT

    photos_dir = os.path.join(root_dir, "Photos")
    videos_dir = os.path.join(root_dir, "Videos")
    review_dir = os.path.join(root_dir, "_AI_REVIEW")

    safe_makedirs(photos_dir)
    safe_makedirs(videos_dir)
    safe_makedirs(review_dir)

    def log(msg: str):
        if log_cb:
            log_cb(msg)

    def progress(done: int, total: int, stage: str):
        if progress_cb:
            progress_cb(done, total, stage)

    def should_stop() -> bool:
        return stop_flag is not None and stop_flag.is_stopped()

    # CSV log file
    log_name = f"{log_prefix}{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_path = os.path.join(root_dir, log_name)

    # Important: do NOT force CLIP load here (keep startup fast).
    # We only report CUDA availability. CLIP device will be logged on first image scoring.
    log(f"Root: {root_dir}")
    log(f"CUDA available: {is_cuda_available()}")
    log("Scanning files...")

    # Scan files
    image_files, video_files, other_files, skipped_files = [], [], [], []

    for root, _, files in os.walk(root_dir):
        if should_stop():
            break

        for f in files:
            if should_stop():
                break

            path = norm_win_path(os.path.join(root, f))

            if skip_target_folders and is_in_target_folders(path, photos_dir, videos_dir, review_dir):
                skipped_files.append(path)
                continue

            ext = os.path.splitext(f)[1].lower()
            if ext in image_ext:
                image_files.append(path)
            elif ext in video_ext:
                video_files.append(path)
            else:
                other_files.append(path)

    stats = {
        "root_dir": root_dir,
        "photos_dir": photos_dir,
        "videos_dir": videos_dir,
        "review_dir": review_dir,
        "csv_log": log_path,
        "dry_run": dry_run,
        "stopped": False,
        "device_hint": get_device_str(),  # not forcing model load
        "images_total": len(image_files),
        "videos_total": len(video_files),
        "others_total": len(other_files),
        "skipped_total": len(skipped_files),
        "screenshots_to_trash": 0,
        "images_to_review": 0,
        "photos_moved": 0,
        "videos_moved": 0,
        "others_to_trash": 0,
        "errors": 0,
    }

    total_work = len(other_files) + len(video_files) + len(image_files)
    done = 0
    progress(done, max(total_work, 1), "start")

    # CSV header
    fieldnames = [
        "timestamp",
        "file_path",
        "file_type",
        "clip_screenshot_score",
        "decision",
        "dest_path",
        "error",
    ]

    # To log the actual CLIP device once, the first time we score an image.
    clip_device_logged = False

    with open(log_path, "w", newline="", encoding="utf-8-sig") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        def log_row(file_path, file_type, score, decision, dest_path="", error=""):
            writer.writerow({
                "timestamp": now_str(),
                "file_path": file_path,
                "file_type": file_type,
                "clip_screenshot_score": "" if score is None else f"{score:.6f}",
                "decision": decision,
                "dest_path": dest_path,
                "error": error,
            })

        # Log skipped files
        for p in skipped_files:
            log_row(p, "skipped", None, "skipped", dest_path=os.path.dirname(p))

        # ----------------------------
        # 1) Other files -> Recycle Bin
        # ----------------------------
        log("Processing non-media files...")
        for path in other_files:
            if should_stop():
                stats["stopped"] = True
                log("Stop requested. Exiting...")
                break

            if not os.path.exists(path):
                stats["errors"] += 1
                log_row(path, "other", None, "missing", error="File not found")
                log(f"SKIP missing file (other): {path}")
                done += 1
                progress(done, max(total_work, 1), "others")
                continue

            try:
                if not dry_run:
                    send2trash(path)
                stats["others_to_trash"] += 1
                log_row(path, "other", None, "other_to_trash", dest_path=("RECYCLE_BIN" if not dry_run else "DRY_RUN"))
            except Exception as e:
                stats["errors"] += 1
                log_row(path, "other", None, "error", error=str(e))
                log(f"ERROR (other): {path} | {e}")

            done += 1
            progress(done, max(total_work, 1), "others")

        # ----------------------------
        # 2) Videos -> Videos/YYYY/YYYY-MM
        # ----------------------------
        if not stats["stopped"]:
            log("Organizing videos...")
            for path in video_files:
                if should_stop():
                    stats["stopped"] = True
                    log("Stop requested. Exiting...")
                    break

                if not os.path.exists(path):
                    stats["errors"] += 1
                    log_row(path, "video", None, "missing", error="File not found")
                    log(f"SKIP missing file (video): {path}")
                    done += 1
                    progress(done, max(total_work, 1), "videos")
                    continue

                try:
                    dt = get_media_date(path, image_ext=image_ext)
                    target_dir = os.path.join(videos_dir, str(dt.year), dt.strftime("%Y-%m"))

                    if dry_run:
                        log_row(path, "video", None, "video_moved", dest_path="DRY_RUN")
                    else:
                        dest = move_file(path, target_dir)
                        log_row(path, "video", None, "video_moved", dest_path=dest)

                    stats["videos_moved"] += 1
                except Exception as e:
                    stats["errors"] += 1
                    log_row(path, "video", None, "error", error=str(e))
                    log(f"ERROR (video): {path} | {e}")

                done += 1
                progress(done, max(total_work, 1), "videos")

        # ----------------------------
        # 3) Images -> CLIP decision
        # ----------------------------
        if not stats["stopped"]:
            log("AI screening images (CLIP GPU)...")
            for path in image_files:
                if should_stop():
                    stats["stopped"] = True
                    log("Stop requested. Exiting...")
                    break

                if not os.path.exists(path):
                    stats["errors"] += 1
                    log_row(path, "image", None, "missing", error="File not found")
                    log(f"SKIP missing file (image): {path}")
                    done += 1
                    progress(done, max(total_work, 1), "images")
                    continue

                try:
                    # On first image scoring, CLIP will lazy-load here.
                    score = clip_screenshot_score(path)

                    if not clip_device_logged:
                        device, _, _, _ = get_clip_bundle()
                        log(f"CLIP initialized on device: {device} (cuda_available={torch.cuda.is_available()})")
                        clip_device_logged = True

                    if score >= screenshot_threshold:
                        if not dry_run:
                            send2trash(path)
                            log_row(path, "image", score, "screenshot_to_trash", dest_path="RECYCLE_BIN")
                        else:
                            log_row(path, "image", score, "screenshot_to_trash", dest_path="DRY_RUN")
                        stats["screenshots_to_trash"] += 1

                    elif score >= review_threshold:
                        if dry_run:
                            log_row(path, "image", score, "review_moved", dest_path="DRY_RUN")
                        else:
                            dest = move_file(path, review_dir)
                            log_row(path, "image", score, "review_moved", dest_path=dest)
                        stats["images_to_review"] += 1

                    else:
                        dt = get_media_date(path, image_ext=image_ext)
                        target_dir = os.path.join(photos_dir, str(dt.year), dt.strftime("%Y-%m"))

                        if dry_run:
                            log_row(path, "image", score, "photo_moved", dest_path="DRY_RUN")
                        else:
                            dest = move_file(path, target_dir)
                            log_row(path, "image", score, "photo_moved", dest_path=dest)
                        stats["photos_moved"] += 1

                except Exception as e:
                    stats["errors"] += 1
                    log_row(path, "image", None, "error", error=str(e))
                    log(f"ERROR (image): {path} | {e}")

                done += 1
                progress(done, max(total_work, 1), "images")

    log(f"Done. CSV log: {log_path}")
    return stats
