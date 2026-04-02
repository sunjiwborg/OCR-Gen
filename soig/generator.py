import os
import random
from PIL import Image, ImageDraw, ImageFont
from .effects import get_background, apply_blur

def generate_image(text, args, filename):
    # Font Selection
    font_path = args.font
    if args.fonts_dir and os.path.exists(args.fonts_dir):
        fonts = [os.path.join(args.fonts_dir, f) for f in os.listdir(args.fonts_dir) 
                 if f.endswith(('.ttf', '.otf'))]
        if fonts:
            font_path = random.choice(fonts)

    font = ImageFont.truetype(font_path, args.pt)

    # Measure Text
    dummy_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = (bbox[2] - bbox[0]) + (args.margin * 2), (bbox[3] - bbox[1]) + (args.margin * 2)

    # Background
    bg_mode = random.choice([0, 1, 2]) if args.rb else args.b
    img = get_background(w, h, bg_mode, "backgrounds")

    # Color
    color = args.color
    if args.rcolor:
        color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))

    # Render Text
    draw = ImageDraw.Draw(img)
    draw.text((args.margin, args.margin), text, font=font, fill=color)

    # Blur Logic
    # If rblur is set, 50% chance to use rblur value, otherwise use standard blur
    final_blur = args.blur
    if args.rblur is not None:
        if random.random() > 0.5:
            final_blur = args.rblur
    
    if final_blur > 0:
        img = apply_blur(img, final_blur)

    # Save
    save_path = os.path.join(args.output, "images", filename)
    img.save(save_path)
    return filename