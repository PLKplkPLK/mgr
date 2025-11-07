import requests
import cv2
import numpy as np
from detector import Detector

def download_and_crop(urls: list, species_name: str, i: int = 1):
    detector = Detector()

    for url in urls:
        if url is np.nan:
            continue
        
        # download
        response = requests.get(url)
        response.raise_for_status()

        image_arr = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
        # crop
        cropped_img, category, _, _, _ = detector.bestBoxDetection(image)

        # save to species/pictures_cropped/
        if cropped_img and category == 1:
            cropped_img.save(f"{species_name}/pictures_cropped/{i}.jpg", format="JPEG")
            i += 1
