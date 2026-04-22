import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image
import math

class GaussianBlur(nn.Module):
    '''
    Adds random noise to a tensor.'''

    def __init__(self, opt=None):
        super(GaussianBlur, self).__init__()
        
        self.name = "G_Blur"

    def get_gaussian_kernel(self, kernel_size, sigma=3, channels=3, device=None):
        
        
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
        if device is not None:
            xy_grid = xy_grid.to(device)  

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        
        
        
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                              (2 * variance)
                          )

        
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        padding = int((kernel_size-1)/2)
        self.gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, padding=padding, groups=channels, bias=False)

        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False
        if device is not None:
            self.gaussian_filter = self.gaussian_filter.to(device)  

        return self.gaussian_filter

    def forward(self, tensor, cur_rate):
        if isinstance(cur_rate, list):
            assert len(cur_rate) == 1
            cur_rate = cur_rate[0]
            
        self.name = "GaussianBlur"
        device = tensor.device  
        channels = tensor.shape[1]  
        
        
        gaussian_layer = self.get_gaussian_kernel(kernel_size=cur_rate, channels=channels, device=device)
        blur_result = gaussian_layer(tensor)
            
            
            
        
        
        return blur_result





















