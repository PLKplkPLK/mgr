import pandas as pd
from PIL import Image

def show_picture(image_path: str) -> Image.Image:
    img = Image.open(image_path)
    img = img.resize((500,500))
    return img

def show_pictures_from_df(df: pd.DataFrame):
    for _, row in df[-150:-145].iterrows():
        image = row['image']
        species = row['detected_animal']
        print(species, end='\r')
        display(show_picture(image))
