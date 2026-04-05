import os
import glob
import random
import numpy as np
from PIL import Image, ImageOps

def generate_background(bg_type_arg, is_random, size, bg_dir, noise_level=15):
    bg_type = bg_type_arg
    if is_random:
        bg_type = random.choice(['0', '1', '2'])

    if bg_type == '1': # Gaussian Noise
        # Create a white base array
        array = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
        # Apply noise based on standard deviation (noise_level)
        noise = np.random.normal(0, noise_level, array.shape)
        noisy_array = np.clip(array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_array)
    
    elif bg_type == '2': # Image Background
        if not bg_dir or not os.path.exists(bg_dir):
            return Image.new('RGB', size, (255, 255, 255))
            
        bg_files = glob.glob(os.path.join(bg_dir, "*.*"))
        if bg_files:
            with Image.open(random.choice(bg_files)) as img:
                # Use .fit to crop/resize without squeezing
                return ImageOps.fit(img.convert("RGB"), size, method=Image.Resampling.LANCZOS)
    
    # Default: Plain White (Type '0')
    return Image.new('RGB', size, (255, 255, 255))