import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
from torch import tensor

DFYOLO_NAME = 'deepfaune-yolov8s_960'
DFYOLO_PATH = '../deepfaune/models/'
DFYOLO_WEIGHTS = DFYOLO_PATH + 'deepfaune-yolov8s_960.pt'

CROP_SIZE = 480 # default 480??
DFYOLO_WIDTH = 960 # image width
DFYOLO_THRES = 0.6
DFYOLOHUMAN_THRES = 0.4 # boxes with human above this threshold are saved
DFYOLOCOUNT_THRES = 0.6

class Detector:
    def __init__(self, name=DFYOLO_NAME, threshold=None, countthreshold=None,
                 humanthreshold=None, dfyolo_weights:str = DFYOLO_WEIGHTS):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {DFYOLO_NAME} with weights at {dfyolo_weights}, in resolution 960x960 on {self.device}")
        self.yolo = YOLO(dfyolo_weights)
        self.yolo.to(self.device)
        self.imgsz = DFYOLO_WIDTH
        self.threshold = DFYOLO_THRES if threshold is None else threshold
        self.countthreshold = DFYOLOCOUNT_THRES if countthreshold is None else countthreshold
        self.humanthreshold = DFYOLOHUMAN_THRES if humanthreshold is None else humanthreshold

    def bestBoxDetection(self, filename_or_imagecv):
        try:
            results = self.yolo(filename_or_imagecv, verbose=False, imgsz=self.imgsz)
        except FileNotFoundError:
            print(f"File '{filename_or_imagecv}' not found")
            return None, 0, np.zeros(4), 0, []
        except Exception as err:
            print(err)
            return None, 0, np.zeros(4), 0, []

        # orig_img a numpy array (cv2) in BGR
        imagecv = results[0].cpu().orig_img
        detection = results[0].cpu().numpy().boxes

        # Are there any relevant boxes?
        if not len(detection.cls) or detection.conf[0] < self.threshold:
            # No. Image considered as empty
            return None, 0, np.zeros(4), 0, []

        # Is there a relevant animal box? 
        try:
            # Yes. Selecting the best animal box
            kbox = np.where((detection.cls==0) & (detection.conf>self.threshold))[0][0]
        except IndexError:
            # No: Selecting the best box for another category (human, vehicle)
            kbox = 0

        # categories are 1=animal, 2=person, 3=vehicle and the empty category 0=empty
        category = int(detection.cls[kbox]) + 1
        box = detection.xyxy[kbox] # xmin, ymin, xmax, ymax

        # Is this an animal box ?
        if category == 1:
            # Yes: cropped image is required for classification
            croppedimage = cropSquareCVtoPIL(imagecv, box.copy())
        else: 
            # No: return none
            return None, 0, np.zeros(4), 0, []

        ## animal count
        # if category == 1:
        #     count = sum((detection.conf>self.countthreshold) & (detection.cls==0)) # only above a threshold
        # else:
        count = 0

        humanboxes = []

        return croppedimage, category, box, count, humanboxes

def cropSquareCVtoPIL(imagecv, box):
    x1, y1, x2, y2 = box
    xsize = (x2-x1)
    ysize = (y2-y1)
    if xsize>ysize:
        y1 = y1-int((xsize-ysize)/2)
        y2 = y2+int((xsize-ysize)/2)
    if ysize>xsize:
        x1 = x1-int((ysize-xsize)/2)
        x2 = x2+int((ysize-xsize)/2)
    height, width, _ = imagecv.shape
    croppedimagecv = imagecv[max(0,int(y1)):min(int(y2),height),max(0,int(x1)):min(int(x2),width)]
    croppedimage = Image.fromarray(croppedimagecv[:,:,(2,1,0)]) # converted to PIL BGR image
    return croppedimage
