import requests
import cv2
import numpy as np
from detector import Detector

def download_and_crop(urls: list, species_name: str, i: int = 1):
    detector = Detector()

    for url in urls:
        if i % 300 == 0:
            print(i)
        if i > 1500:
            break

        if url is np.nan:
            continue

        url.replace('medium', 'large')
        
        # download
        response = requests.get(url)
        try:
            response.raise_for_status()
        except:
            print(f'File missing: {url}')
            continue
        image_arr = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
        # crop
        cropped_img, category, _, _, _ = detector.bestBoxDetection(image)

        # save to species/pictures_cropped/
        if cropped_img is not None and category == 1 and cropped_img.size[0] > 120:
            cropped_img.save(f"{species_name}/pictures_cropped/ina_{i}.jpg", format="JPEG")
            i += 1
