import os
import glob
import random
from PIL import ImageFont

def load_dictionary(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def get_font(args):
    # Determine size
    font_size = args.pt
    if args.rpt:
        font_size = random.randint(args.rpt[0], args.rpt[1])

    font_path = args.font
    if args.font_dir:
        font_files = glob.glob(os.path.join(args.font_dir, "*.[to]tf"))
        if font_files:
            font_path = random.choice(font_files)
    
    try:
        # layout_engine=RAQM enables complex script support
        return ImageFont.truetype(font_path, size=font_size, layout_engine=ImageFont.Layout.RAQM)
    except Exception:
        return ImageFont.load_default(size=font_size)