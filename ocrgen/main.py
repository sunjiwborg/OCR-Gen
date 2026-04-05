import argparse
import os
import random
from tqdm import tqdm
from PIL import Image, ImageDraw
from .text_utils import load_dictionary, get_font
from .image_utils import apply_rotation_to_layer, apply_blur_to_final, get_font_color
from .bg_utils import generate_background

def main():
    parser = argparse.ArgumentParser(description="OCR-Gen: Synthetic Text Image Generator")
    # Core settings
    parser.add_argument("-lang", required=True, help="Path to dictionary (.txt)")
    parser.add_argument("-w", type=int, default=3, help="Words per sample")
    parser.add_argument("-c", type=int, default=10, help="Total samples")
    parser.add_argument("-out_dir", required=True, help="Output directory")
    
    # Text Appearance
    parser.add_argument("-pt", type=int, default=32, help="Font point size")
    parser.add_argument("-rpt", type=int, nargs=2, metavar=('MIN', 'MAX'), help="Random point size range")
    parser.add_argument("-color", default="#000000", help="Font hex color")
    parser.add_argument("-rcolor", action="store_true", help="Random font color")
    parser.add_argument("-margin", type=int, default=10, help="Margin around text")
    
    # Backgrounds
    parser.add_argument("-b", type=str, choices=['0', '1', '2'], default='0', help="0:White, 1:Noise, 2:Image")
    parser.add_argument("-rb", action="store_true", help="Randomize background type")
    parser.add_argument("-bg_dir", default="backgrounds", help="Folder for -b 2")
    parser.add_argument("-noise", type=float, default=15.0, help="Gaussian noise intensity")
    
    # Effects
    parser.add_argument("-t", type=float, help="Tilt/Rotation angle")
    parser.add_argument("-rt", action="store_true", help="Random tilt (-10 to 10)")
    parser.add_argument("-bl", type=float, help="Gaussian blur radius")
    parser.add_argument("-rbl", action="store_true", help="Random blur (0, 1, 2)")
    
    # Font Source
    parser.add_argument("-font", help="Specific font path")
    parser.add_argument("-font_dir", help="Folder for random font selection")
    
    args = parser.parse_args()

    # Setup output
    img_dir = os.path.join(args.out_dir, "images_folder")
    os.makedirs(img_dir, exist_ok=True)
    words = load_dictionary(args.lang)

    map_path = os.path.join(args.out_dir, "map.tsv")
    with open(map_path, 'w', encoding='utf-8') as tsv:
        for i in tqdm(range(args.c), desc="Generating Images", unit="img"):
            text = " ".join(random.sample(words, min(len(words), args.w)))
            font = get_font(args)
            
            # 1. Measure Text
            temp_draw = ImageDraw.Draw(Image.new('RGBA', (1,1)))
            bbox = temp_draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

            # 2. Render Text Layer
            text_layer = Image.new('RGBA', (tw + 10, th + 10), (0, 0, 0, 0))
            tdraw = ImageDraw.Draw(text_layer)
            tdraw.text((-bbox[0] + 5, -bbox[1] + 5), text, font=font, fill=get_font_color(args))

            # 3. Apply Tilt
            if args.rt or args.t is not None:
                angle = args.t if not args.rt else random.uniform(-10, 10)
                text_layer = apply_rotation_to_layer(text_layer, angle)

            # 4. Canvas & Background
            lw, lh = text_layer.size
            canvas_size = (lw + (args.margin * 2), lh + (args.margin * 2))
            bg_img = generate_background(args.b, args.rb, canvas_size, args.bg_dir, args.noise)

            # 5. Composite & Effects
            bg_img.paste(text_layer, (args.margin, args.margin), text_layer)
            final_img = apply_blur_to_final(bg_img, args)
            
            # 6. Save
            fname = f"sample_{i}.png"
            final_img.save(os.path.join(img_dir, fname))
            tsv.write(f"{fname}\t{text}\n")

    print(f"\nDone! Dataset saved to: {args.out_dir}")

if __name__ == "__main__":
    main()