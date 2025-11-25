from PIL import Image

def show_picture(image_path: str) -> Image.Image:
    return Image.open(image_path)
