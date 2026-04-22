import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image

class Resize(nn.Module):
    """
    Resize the image. The target size is original size * resize_ratio
    """
    def __init__(self, interpolation_method='nearest'):
        super(Resize, self).__init__()
        self.interpolation_method = interpolation_method

    def forward(self, img, scale_factor):
        '''缩小到scale_factor再放大回原样'''
        noised_image = img
        image_size = noised_image.shape[-2:]
        noised_image = F.interpolate(
                                    noised_image,
                                    scale_factor=(scale_factor, scale_factor),
                                    mode=self.interpolation_method)
        noised_image = F.interpolate(
                                    noised_image,
                                    size=tuple(image_size),
                                    mode=self.interpolation_method)
        return noised_image