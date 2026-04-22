import torch
import torch.nn as nn
import numpy as np
from attack.JpegCompression import JpegFASL
from attack.Crop import Crop
from attack.Resize import Resize
from attack.GaussianNoise import GaussianNoise

class AttackLayer(nn.Module):
    def __init__(self, opts):
        super(AttackLayer, self).__init__()
        self.jpeg_layer = JpegFASL()
        self.crop_layer = Crop()
        self.resize_layer = Resize()
        self.gaussian_layer = GaussianNoise()

        self.jpeg_min = opts['JPEG'][0]
        self.jpeg_max = opts['JPEG'][1]

        self.crop_min = opts['CROP'][0]
        self.crop_max = opts['CROP'][1]

        self.resize_min = opts['RESIZE'][0]
        self.resize_max = opts['RESIZE'][1]

        self.Gaussian_var = opts['GAUSSIAN']

    
    def forward(self, image_tensor):
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        prob = np.random.rand()
        if prob < 0.5:
            attacked = image_tensor  
        else:
            jpeg_quality = np.random.randint(self.jpeg_min, self.jpeg_max)  
            
            attacked = self.gaussian_layer(image_tensor)
            attacked, _, _ = self.jpeg_layer(attacked, jpeg_quality)
            
        return attacked


