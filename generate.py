import os
import random
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
from tqdm import tqdm

INPUT_PARQUET = "bodo_text_data.parquet"
OUTPUT_DIR = "bodo_ocr_dataset"
FONTS_DIR = "fonts"             
TARGET_COUNT = 10000            
MAX_CHARS_PER_LINE = 45         
TRAIN_SPLIT = 0.8              

def add_salt_pepper(img_array, prob=0.015):
    thres = 1 - prob
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            rdn = random.random()
            if rdn < prob: img_array[i][j] = 0
            elif rdn > thres: img_array[i][j] = 255
    return img_array

def apply_stress(img):
    img_array = np.array(img.convert('L'))
    
    choice = random.choice(['sp', 'blur', 'ink', 'none'])
    if choice == 'sp':
        img_array = add_salt_pepper(img_array)
        img = Image.fromarray(img_array)
    elif choice == 'blur':
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
    elif choice == 'ink':
        img = img.filter(ImageFilter.MinFilter(3)) 

    img = img.rotate(random.uniform(-2.0, 2.0), resample=Image.BICUBIC, expand=False, fillcolor="white")

    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    return img

def generate():
    df = pd.read_parquet(INPUT_PARQUET)
    font_files = [f for f in os.listdir(FONTS_DIR) if f.endswith(('.ttf', '.otf'))]
    
    if not font_files:
        print("Error: Put .ttf files in the /fonts folder first!")
        return

    for sub in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

    success_count = 0
    manifest = []
    
    pbar = tqdm(total=TARGET_COUNT, desc="Generating Bodo OCR Data")
    
    df = df.sample(frac=1).reset_index(drop=True)

    for i, row in df.iterrows():
        if success_count >= TARGET_COUNT:
            break
            
        text = str(row['text']).strip()[:MAX_CHARS_PER_LINE]
        if not text or len(text) < 3: 
            continue

        font_path = os.path.join(FONTS_DIR, random.choice(font_files))
        font = ImageFont.truetype(font_path, random.randint(28, 38))

        bbox = ImageDraw.Draw(Image.new('L', (1, 1))).textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        img = Image.new('RGB', (w + 60, h + 60), "white")
        draw = ImageDraw.Draw(img)
        draw.text((30, 30), text, font=font, fill="black")

        img = apply_stress(img)

        folder = "train" if random.random() < TRAIN_SPLIT else "val"
        filename = f"bodo_{success_count:06d}.png"
        save_path = os.path.join(OUTPUT_DIR, folder, filename)
        
        img.save(save_path)
        with open(f"{save_path}.gt.txt", "w", encoding="utf-8") as f:
            f.write(text)

        manifest.append({"file_name": filename, "text": text, "split": folder})
        success_count += 1
        pbar.update(1)

    pbar.close()
    pd.DataFrame(manifest).to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)
    print(f"\nFinished! Your dataset is ready in: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate()

