import random

import numpy as np
import torch
from PIL import Image


def resize_image_and_mask(image, mask, image_size=None):
    if image_size is None:
        return image, mask
    size = (image_size, image_size)
    image = image.resize(size, Image.Resampling.BILINEAR)
    mask = mask.resize(size, Image.Resampling.NEAREST)
    return image, mask


def image_to_tensor(image, center=True):
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1))
    if center:
        tensor = tensor - 0.5
    return tensor


def mask_to_tensor(mask):
    array = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0)


class ImageMaskTransform:
    def __init__(self,
                 train=False,
                 center=True,
                 horizontal_flip_prob=0.0,
                 image_size=180):
        self.train = train
        self.center = center
        self.horizontal_flip_prob = horizontal_flip_prob
        self.image_size = image_size

    def __call__(self, image, mask):
        image, mask = resize_image_and_mask(image, mask, image_size=self.image_size)
        image_tensor = image_to_tensor(image, center=self.center)
        mask_tensor = mask_to_tensor(mask)
        if self.train and self.horizontal_flip_prob > 0.0:
            if random.random() < self.horizontal_flip_prob:
                image_tensor = torch.flip(image_tensor, dims=[2])
                mask_tensor = torch.flip(mask_tensor, dims=[2])
        return image_tensor, mask_tensor
