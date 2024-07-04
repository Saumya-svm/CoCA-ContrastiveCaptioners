from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass
import pandas as pd
from PIL import Image
import numpy as np
from typing import Tuple

def img2array(file):
    img = Image.open(file)
    arr = np.array(img)
    return arr

class flickr(Dataset):
    def __init__(self) -> None:
        super(flickr, self).__init__()

        df = pd.read_csv("final.csv")
        unique_objects = df['image'].unique()
        self.imgs = unique_objects
        self.captions = df['caption']

    def __getitem__(self, index) -> Tuple[np.ndarray, str]:
        img = img2array(self.imgs[index//5])
        caption = self.captions[index]
        return img, caption
    
    def __len__(self):
        return len(self.captions)

def get_data():
    pass