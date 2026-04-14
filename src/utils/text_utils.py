import os
import random
import glob
from tqdm import tqdm
from PIL import ImageFont

# Read texts from source file
def reader(path):
    with open(path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in tqdm(f, total=total_lines, desc="Reading file...", ascii=True) if line.strip()]

def get_font(args):
    # Determine font size
    font_size = random.choice(args.pt)
    # Finding available fonts
    font_files = glob.glob(os.path(args.f, "*.[to]tf"))

    if font_files:
        font_path = random.choice(font_files)
    else:
        print("Couldn't find fonts")
        exit
    
    try:
        # layout_engine=ImageFont.Layout.RAQM enables complex script support
        return ImageFont.truetype(font_path, size=font_size, layout_engine=ImageFont.Layout.RAQM)
    except Exception:
        return ImageFont.load_default(size=font_size)

