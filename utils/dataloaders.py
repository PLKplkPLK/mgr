import ast
import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms.functional import pad

from utils.cropping import crop_normalized_bbox_square
from utils.class_names import class_to_idx, class_names


IMG_SIZE = 480
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_dim = max(w, h)

        pad_left   = (max_dim - w) // 2
        pad_right  = max_dim - w - pad_left
        pad_top    = (max_dim - h) // 2
        pad_bottom = max_dim - h - pad_top

        return pad(
            image,
            [pad_left, pad_top, pad_right, pad_bottom],
            padding_mode='reflect'
        )


class CroppedImageDataset(Dataset):
    def __init__(self, df, img_size: int):
        # keep only a fraction of rows (randomly)
        self.df = df
        self.transform = transforms.Compose([
            SquarePad(),
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        species = row["species"]
        label = class_to_idx[species]
        bbox = ast.literal_eval(row["bbox"])

        # Load image
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # choose a random other index and try again
            new_idx = np.random.randint(0, len(self))
            return self.__getitem__(new_idx)
        img = crop_normalized_bbox_square(img, bbox)

        # Transform (resize, ToTensor, etc.)
        if self.transform:
            img = self.transform(img)

        return img, label

def is_valid_image(path):
    dirname = os.path.basename(os.path.dirname(path))
    return dirname == "pictures_cropped" and path.lower().endswith(VALID_EXT)

def get_dataloaders(batch_size: int, img_size: int = IMG_SIZE) -> dict:
    """
    Returns dictionary with train_dataloader, test_dataloader,
    train_dataset and test_dataset.
    """
    ### TRAIN ###
    training_data_transforms = transforms.Compose([
        SquarePad(),
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.ImageFolder(
        root="../../data",
        transform=training_data_transforms,
        is_valid_file=is_valid_image,
    )
    print('Number of train classes: ', len(train_dataset.classes))
    print('Number of train images: ', len(train_dataset))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    if (train_dataset.class_to_idx != class_to_idx):
        raise ValueError('Mapping doesn\'t match')

    ### TEST ###
    df = pd.read_csv('../../y_clean_thin.csv', index_col=0).sample(frac=0.2).reset_index(drop=True)
    df_mega = pd.read_csv('../../megadetector_results.csv', index_col=0)
    df = df.merge(df_mega, on='image_path')
    df.image_path = '../../' + df.image_path
    df = df.dropna()
    df = df[df["species"].isin(class_names)].reset_index(drop=True)

    test_dataset = CroppedImageDataset(df, img_size)
    print('Number of test images: ', len(test_dataset))

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    return {
        'train_dataloader': train_dataloader,
        'test_dataloader': test_dataloader,
        'train_dataset': train_dataset,
        'test_dataset': test_dataset
    }
