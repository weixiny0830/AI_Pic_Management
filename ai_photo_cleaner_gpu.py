import os
import shutil
import torch
import clip
from PIL import Image, ExifTags
from datetime import datetime
from send2trash import send2trash
from tqdm import tqdm
import pillow_heif

# ================== 配置 ==================
ROOT_DIR = r"D:\Photos"   # 👈 改成你的图片目录
REVIEW_DIR = os.path.join(ROOT_DIR, "_AI_REVIEW")

SCREENSHOT_THRESHOLD = 0.80
REVIEW_THRESHOLD = 0.60

IMAGE_EXT = ('.jpg', '.jpeg', '.png', '.heic')

# =========================================

pillow_heif.register_heif_opener()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

TEXTS = [
    "a smartphone screenshot",
    "a chat app interface",
    "a webpage screenshot",
    "a document screenshot",
    "a computer screen",
    "a real-life photograph",
    "a family photo",
    "a landscape photo"
]

text_tokens = clip.tokenize(TEXTS).to(device)

def clip_score(image_path):
    try:
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            image_feat = model.encode_image(image)
            text_feat = model.encode_text(text_tokens)
            probs = (image_feat @ text_feat.T).softmax(dim=-1)[0]
        screenshot_score = probs[:5].sum().item()
        photo_score = probs[5:].sum().item()
        return screenshot_score, photo_score
    except:
        return 0, 1

def get_photo_date(path):
    try:
        img = Image.open(path)
        exif = img._getexif()
        if exif:
            for tag, val in exif.items():
                if ExifTags.TAGS.get(tag) == "DateTimeOriginal":
                    return datetime.strptime(val, "%Y:%m:%d %H:%M:%S")
    except:
        pass
    return datetime.fromtimestamp(os.path.getctime(path))

# 收集图片
all_images = []
for root, _, files in os.walk(ROOT_DIR):
    if root.startswith(REVIEW_DIR):
        continue
    for f in files:
        if f.lower().endswith(IMAGE_EXT):
            all_images.append(os.path.join(root, f))

os.makedirs(REVIEW_DIR, exist_ok=True)

stats = {"screenshots": 0, "photos": 0, "review": 0}

for path in tqdm(all_images, desc="AI Processing"):
    s_score, p_score = clip_score(path)

    if s_score >= SCREENSHOT_THRESHOLD:
        send2trash(path)
        stats["screenshots"] += 1
        continue

    if s_score >= REVIEW_THRESHOLD:
        shutil.move(path, os.path.join(REVIEW_DIR, os.path.basename(path)))
        stats["review"] += 1
        continue

    # 正常照片 → 按月分类
    date = get_photo_date(path)
    target_dir = os.path.join(
        ROOT_DIR,
        str(date.year),
        date.strftime("%Y-%m")
    )
    os.makedirs(target_dir, exist_ok=True)

    target_path = os.path.join(target_dir, os.path.basename(path))
    if not os.path.exists(target_path):
        shutil.move(path, target_path)

    stats["photos"] += 1

print("\n===== 处理完成 =====")
for k, v in stats.items():
    print(f"{k}: {v}")
