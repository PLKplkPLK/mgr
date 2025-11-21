import os
import sys

import pandas as pd
import numpy as np
import torch
import timm
import torch.nn as nn
# from ultralytics import YOLO
from torch import tensor
from torchvision.transforms import transforms, InterpolationMode


CROP_SIZE = 480
BACKBONE = "vit_large_patch14_dinov2.lvd142m"
class_names = [
        'bison', 'badger', 'ibex', 'beaver', 'red deer', 'golden jackal',
        'chamois', 'cat', 'goat', 'roe deer', 'dog', 'raccoon dog',
        'fallow deer', 'squirrel', 'moose', 'equid', 'genet', 'wolverine',
        'hedgehog', 'lagomorph', 'wolf', 'otter', 'lynx', 'marmot',
        'micromammal', 'mouflon', 'sheep', 'mustelid', 'bird', 'bear',
        'porcupine', 'nutria', 'muskrat', 'raccoon', 'fox', 'reindeer',
        'wild boar', 'cow']

class Deepfaune:
    def __init__(self, dfvit_weights):
        self.model = Model()
        self.model.loadWeights(dfvit_weights)
        # transform image to form usable by network
        self.transforms = transforms.Compose([
            transforms.Resize(
                size=(CROP_SIZE, CROP_SIZE),
                interpolation=InterpolationMode.BICUBIC,
                max_size=None, antialias=None),
            transforms.ToTensor(),
            transforms.Normalize(mean=tensor([0.4850, 0.4560, 0.4060]),
                                 std=tensor([0.2290, 0.2240, 0.2250]))])

    def predictOnBatch(self, batchtensor, withsoftmax=True):
        return self.model.predict(batchtensor, withsoftmax)

    # croppedimage loaded by PIL
    def preprocessImage(self, croppedimage):
        preprocessimage = self.transforms(croppedimage)
        return preprocessimage.unsqueeze(dim=0)


class Model(nn.Module):
    def __init__(self):
        """
        Constructor of model classifier
        """
        super().__init__()
        self.base_model = timm.create_model(BACKBONE, pretrained=False, num_classes=len(class_names),
                                            dynamic_img_size=True)
        print(f"Using model in resolution {CROP_SIZE}x{CROP_SIZE}")
        self.backbone = BACKBONE
        self.nbclasses = len(class_names)

    def forward(self, input):
        x = self.base_model(input)
        return x

    def predict(self, data, withsoftmax=True):
        """
        Predict on test DataLoader
        :param test_loader: test dataloader: torch.utils.data.DataLoader
        :return: numpy array of predictions without soft max
        """
        self.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        total_output = []
        with torch.no_grad():
            x = data.to(device)
            if withsoftmax:
                output = self.forward(x).softmax(dim=1)
            else:
                output = self.forward(x)
            total_output += output.tolist()

        return np.array(total_output)

    def loadWeights(self, path):
        """
        :param path: path of .pt save of model
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("CUDA available" if torch.cuda.is_available() else "CUDA unavailable. Using CPU")

        try:
            params = torch.load(path, map_location=device, weights_only=False)
            args = params['args']
            if self.nbclasses != args['num_classes']:
                raise Exception("You load a model ({}) that does not have the same number of class"
                                "({})".format(args['num_classes'], self.nbclasses))
            self.backbone = args['backbone']
            self.nbclasses = args['num_classes']
            self.load_state_dict(params['state_dict'])
        except Exception as e:
            print("Can't load checkpoint model because :\n\n " + str(e), file=sys.stderr)
            raise e


def get_model(dfvit_weights: str = 'deepfaune/models/deepfaune-vit_large_patch14_dinov2.lvd142m.v4.pt') -> tuple[Deepfaune, list[str]]:
    """Return Deepfaune model and class names."""

    model = Deepfaune(dfvit_weights)
    return model, class_names
