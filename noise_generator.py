import os
import csv
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

# ── CONFIG ────────────────────────────────────────────────
CLEAN_DIR   = "data/clean"
NOISY_DIR   = "data/noisy"
LABELS_FILE = "data/labels.csv"
NOISY_COUNT = 1000              
# ─────────────────────────────────────────────────────────

os.makedirs(NOISY_DIR, exist_ok=True)

# ── Load clean labels ─────────────────────────────────────
clean_records = []
with open(LABELS_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        clean_records.append((row["image_path"], row["text"]))

# ── Pick random 1500 to noisify ───────────────────────────
random.shuffle(clean_records)
selected = clean_records[:NOISY_COUNT]

# ── Noise functions ───────────────────────────────────────
def add_blur(img):
    radius = random.uniform(0.8, 1.5)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def add_brightness(img):
    factor = random.uniform(0.3, 0.6)
    return ImageEnhance.Brightness(img).enhance(factor)

def add_rotation(img):
    angle = random.uniform(-3, 3)
    return img.rotate(angle, fillcolor="white", expand=False)

def add_gaussian_noise(img):
    arr   = np.array(img).astype(np.int16)
    noise = np.random.randint(0, 100, arr.shape, dtype=np.int16)
    arr   = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def add_contrast(img):
    factor = random.uniform(1, 1.5)
    return ImageEnhance.Contrast(img).enhance(factor)

def add_sharpness(img):
    factor = random.uniform(0.5, 2.0)
    return ImageEnhance.Sharpness(img).enhance(factor)

def add_salt_pepper_noise(img, amount=0.01):
    arr    = np.array(img).astype(np.uint8)
    total  = arr.size // 3  # total pixels

    # Salt (white pixels)
    num_salt = int(total * amount)
    coords   = [np.random.randint(0, i, num_salt) for i in arr.shape[:2]]
    arr[coords[0], coords[1]] = 255

    # Pepper (black pixels)
    num_pepper = int(total * amount)
    coords     = [np.random.randint(0, i, num_pepper) for i in arr.shape[:2]]
    arr[coords[0], coords[1]] = 0

    return Image.fromarray(arr)

AUGMENTATIONS = [
    add_blur,
    add_brightness,
    add_rotation,
    add_gaussian_noise,
    add_contrast,
    add_sharpness,
    add_salt_pepper_noise
]

def apply_noise(img):
    # Apply 1–2 random augmentations
    effects = random.sample(AUGMENTATIONS, k=random.randint(1, 2))
    for effect in effects:
        img = effect(img)
    return img

# ── Generate noisy images ─────────────────────────────────
print("Generating noisy images...")
noisy_records = []

for i, (img_path, text) in enumerate(selected):
    try:
        img       = Image.open(img_path).convert("RGB")
        noisy_img = apply_noise(img)

        fname     = os.path.basename(img_path).replace(".png", "_noisy.png")
        out_path  = os.path.join(NOISY_DIR, fname)
        noisy_img.save(out_path)
        noisy_records.append((out_path, text))

        if i % 500 == 0:
            print(f"  {i} noisy images done...")

    except Exception as e:
        print(f"  Skipped: {e}")
        continue

# ── Append noisy labels to CSV ────────────────────────────
with open(LABELS_FILE, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for img_path, label in noisy_records:
        writer.writerow([img_path, label])

print(f"\n✅ Done! {len(noisy_records)} noisy images saved to '{NOISY_DIR}'")
print(f"✅ Noisy labels appended to '{LABELS_FILE}'")
print(f"✅ Total dataset: {len(clean_records) + len(noisy_records)} pairs")
