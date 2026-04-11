from torch.utils.data import Dataset
import torch
import os
from glob import glob
import cv2
import numpy as np
# from PIL import Image
from pathlib import Path
import json

class UnetDataset(Dataset):
    def __init__(self, dataset_dir, transforms=None):
        self.image_paths = glob(os.path.join(dataset_dir, "*.bmp"))
        self.transforms = transforms
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        # image = cv2.resize(image, (128,128))
        mask = self.get_mask(self.image_paths[idx], image)

        if self.transforms:
            image = self.transforms(image) # [540, 720] --> [1, 540, 720]
            mask = self.transforms(mask)
        return image, mask

    def get_mask(self, image_path, image):
        json_path = image_path.replace(".bmp", ".json")
        points = self.get_points(json_path)

        mask = np.zeros_like(image)
        mask = cv2.fillPoly(mask, [points], color=255)

        return mask

    def get_points(self, json_path):
        with open(json_path, 'r') as jsonfile:
            data = json.load(jsonfile)
        shapes = data['shapes']
        for shape in shapes:
            points = shape['points']
            points = np.array(points).round().astype(int)
        return points
        

# dataset = UnetDataset(dataset_dir=os.path.join('sample_dataset', 'test'))
# # for idx in range(len(dataset)):
# idx = 0
# shuffled_idx = np.arange(len(dataset))
# np.random.shuffle(shuffled_idx)
# while True:
#     image, mask = dataset[shuffled_idx[idx]]
#     print(f"{idx}/{len(dataset)-1}")
#     # print(mask)
#     image_mask = np.hstack((image, mask))
#     cv2.imshow('image', image_mask)
#     key = cv2.waitKey()

#     if key==ord('q'):
#         break
#     elif key==ord('a'):
#         idx -= 1
#     elif key==ord('d'):
#         idx += 1
#     if idx < 0:
#         idx = 0
#     if idx > len(dataset)-1:
#         idx = len(dataset)
    
