import argparse
import os
import csv
from tqdm import tqdm
from .generator import generate_image

def main():
    parser = argparse.ArgumentParser(description="Synthetic Text Image Generator CLI")
    
    # Arguments
    parser.add_argument("--text", required=True, help="Input .txt file")
    parser.add_argument("--output", default="output", help="Root output directory")
    parser.add_argument("--font", default="arial.ttf", help="Path to default font")
    parser.add_argument("--pt", type=int, default=32, help="Font size")
    parser.add_argument("--color", default="black", help="Text color")
    parser.add_argument("--blur", type=float, default=0, help="Standard blur")
    parser.add_argument("--b", type=int, choices=[0, 1, 2], default=0, help="BG effect")
    parser.add_argument("--margin", type=int, default=10, help="Text margin")
    parser.add_argument("--fonts_dir", help="Directory of fonts")
    parser.add_argument("--rblur", type=float, help="Random blur amount")
    parser.add_argument("--rcolor", action="store_true", help="Randomize text color")
    parser.add_argument("--rb", action="store_true", help="Randomize background")

    args = parser.parse_args()

    # Directory Setup
    img_dir = os.path.join(args.output, "images")
    os.makedirs(img_dir, exist_ok=True)
    tsv_path = os.path.join(args.output, "mapping.tsv")

    if not os.path.exists(args.text):
        print(f"Error: {args.text} not found.")
        return

    # Read lines
    with open(args.text, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Main Loop with Progress Bar
    with open(tsv_path, 'w', encoding='utf-8', newline='') as tsv_file:
        writer = csv.writer(tsv_file, delimiter='\t')
        
        for i, line in enumerate(tqdm(lines, desc="Generating Images", unit="img")):
            filename = f"line_{i:05d}.png"
            generate_image(line, args, filename)
            writer.writerow([filename, line])

    print(f"\nSuccess! Processed {len(lines)} lines.")
    print(f"Data saved to: {args.output}")

if __name__ == "__main__":
    main()