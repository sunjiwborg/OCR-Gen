import numpy as np
import random
import os
from PIL import Image, ImageFilter

def get_background(width, height, mode, bg_folder=None):
    if mode == 1:  # Gaussian Noise
        noise = np.random.normal(220, 30, (height, width, 3))
        noise = np.clip(noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noise)
    
    elif mode == 2 and bg_folder and os.path.exists(bg_folder): # Random Image
        bgs = [f for f in os.listdir(bg_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if bgs:
            bg_path = os.path.join(bg_folder, random.choice(bgs))
            img = Image.open(bg_path).convert("RGB")
            return img.resize((width, height), Image.Resampling.LANCZOS)
            
    # Default: Plain White
    return Image.new("RGB", (width, height), (255, 255, 255))

def apply_blur(image, amount):
    return image.filter(ImageFilter.GaussianBlur(radius=amount))