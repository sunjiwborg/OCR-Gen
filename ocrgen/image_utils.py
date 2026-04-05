import random
from PIL import Image, ImageFilter

def apply_rotation_to_layer(text_layer, angle):
    """Rotates the text layer (tilt) and expands the canvas to fit the slanted text."""
    return text_layer.rotate(angle, resample=Image.BICUBIC, expand=True)

def apply_blur_to_final(image, args):
    if args.rbl:
        radius = random.choice([0, 1, 2])
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    elif args.bl is not None:
        return image.filter(ImageFilter.GaussianBlur(radius=args.bl))
    return image

def get_font_color(args):
    if args.rcolor:
        return (random.randint(0, 150), random.randint(0, 150), random.randint(0, 150))
    return args.color