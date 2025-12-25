import os
import csv
import shutil
from datetime import datetime

import torch
import clip
from PIL import Image, ExifTags
from send2trash import send2trash
from tqdm import tqdm
import pillow_heif

# ================== CONFIGURATION (only change ROOT_DIR) ==================
ROOT_DIR = r"D:\Pictures"   # Change to your root directory

# Confidence thresholds
SCREENSHOT_THRESHOLD = 0.80   # >= 0.80 → screenshot → Recycle Bin
REVIEW_THRESHOLD = 0.60       # 0.60–0.80 → _AI_REVIEW

# File extensions
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.heic', '.gif'}
VIDEO_EXT = {'.mp4', '.mov', '.avi', '.mkv', '.wmv', '.mts', '.m2ts'}

# Output folders (created automatically under ROOT_DIR)
PHOTOS_DIR = os.path.join(ROOT_DIR, "Photos")
VIDEOS_DIR = os.path.join(ROOT_DIR, "Videos")
REVIEW_DIR = os.path.join(ROOT_DIR, "_AI_REVIEW")

# Skip files already inside target folders
SKIP_TARGET_FOLDERS = True

# CSV log
LOG_PREFIX = "_ai_organizer_log_"
# ========================================================================

# Enable HEIC support
pillow_heif.register_heif_opener()

# Initialize CLIP (GPU preferred)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

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
text_tokens = clip.tokenize(TEXTS).to(device)

EXIF_TAGS = ExifTags.TAGS


def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_media_date(path: str) -> datetime:
    """
    Try EXIF DateTimeOriginal for images.
    Fallback to Windows file creation time.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXT:
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
        img = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(img).unsqueeze(0).to(device)

        image_feat = model.encode_image(image_tensor)
        text_feat = model.encode_text(text_tokens)

        probs = (image_feat @ text_feat.T).softmax(dim=-1)[0]
        return float(probs[:5].sum().item())
    except Exception:
        # If image is corrupted/unreadable, treat as non-screenshot
        return 0.0


def in_target_folders(path: str) -> bool:
    if not SKIP_TARGET_FOLDERS:
        return False

    norm = os.path.normpath(path)
    return (
        norm.startswith(os.path.normpath(PHOTOS_DIR) + os.sep)
        or norm.startswith(os.path.normpath(VIDEOS_DIR) + os.sep)
        or norm.startswith(os.path.normpath(REVIEW_DIR) + os.sep)
    )


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


def main():
    safe_makedirs(PHOTOS_DIR)
    safe_makedirs(VIDEOS_DIR)
    safe_makedirs(REVIEW_DIR)

    # CSV log file
    log_name = f"{LOG_PREFIX}{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_path = os.path.join(ROOT_DIR, log_name)

    # Scan files
    image_files, video_files, other_files, skipped_files = [], [], [], []

    for root, _, files in os.walk(ROOT_DIR):
        for f in files:
            path = os.path.join(root, f)

            if in_target_folders(path):
                skipped_files.append(path)
                continue

            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXT:
                image_files.append(path)
            elif ext in VIDEO_EXT:
                video_files.append(path)
            else:
                other_files.append(path)

    stats = {
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

        # Other files → Recycle Bin
        for path in tqdm(other_files, desc="Sending non-media files to Recycle Bin"):
            try:
                send2trash(path)
                stats["others_to_trash"] += 1
                log_row(path, "other", None, "other_to_trash", dest_path="RECYCLE_BIN")
            except Exception as e:
                stats["errors"] += 1
                log_row(path, "other", None, "error", error=str(e))

        # Videos → Videos/YYYY/YYYY-MM
        for path in tqdm(video_files, desc="Organizing videos"):
            try:
                dt = get_media_date(path)
                target_dir = os.path.join(VIDEOS_DIR, str(dt.year), dt.strftime("%Y-%m"))
                dest = move_file(path, target_dir)
                stats["videos_moved"] += 1
                log_row(path, "video", None, "video_moved", dest_path=dest)
            except Exception as e:
                stats["errors"] += 1
                log_row(path, "video", None, "error", error=str(e))

        # Images → CLIP decision
        for path in tqdm(image_files, desc="AI screening images (CLIP GPU)"):
            try:
                score = clip_screenshot_score(path)

                if score >= SCREENSHOT_THRESHOLD:
                    send2trash(path)
                    stats["screenshots_to_trash"] += 1
                    log_row(path, "image", score, "screenshot_to_trash", dest_path="RECYCLE_BIN")
                    continue

                if score >= REVIEW_THRESHOLD:
                    dest = move_file(path, REVIEW_DIR)
                    stats["images_to_review"] += 1
                    log_row(path, "image", score, "review_moved", dest_path=dest)
                    continue

                dt = get_media_date(path)
                target_dir = os.path.join(PHOTOS_DIR, str(dt.year), dt.strftime("%Y-%m"))
                dest = move_file(path, target_dir)
                stats["photos_moved"] += 1
                log_row(path, "image", score, "photo_moved", dest_path=dest)

            except Exception as e:
                stats["errors"] += 1
                log_row(path, "image", None, "error", error=str(e))

    print("\n================= DONE =================")
    print(f"CLIP device: {device} (cuda_available={torch.cuda.is_available()})")
    print("CSV log:", log_path)
    for k, v in stats.items():
        print(f"{k}: {v}")
    print("Review folder:", REVIEW_DIR)


if __name__ == "__main__":
    main()