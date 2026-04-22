import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image

class Crop(nn.Module):
	"""
	Crop randomly sized images and rescale to original size
	"""
	def __init__(self):
		super(Crop, self).__init__()

	def get_random_rectangle_inside(self, image_shape, height_ratio, width_ratio):
		'''返回随机的长方形区域(返回顶点坐标)'''
               
		image_height = image_shape[2]
		image_width = image_shape[3]

		remaining_height = int(height_ratio * image_height)
		remaining_width = int(width_ratio * image_width)

           
		if remaining_height == image_height:
			height_start = 0
		else:
			height_start = np.random.randint(0, image_height - remaining_height)

		if remaining_width == image_width:
			width_start = 0
		else:
			width_start = np.random.randint(0, image_width - remaining_width)

		return height_start, height_start + remaining_height, width_start, width_start + remaining_width

	def forward(self, image, height_ratio, width_ratio, apex=None):
              
		height_ratio = min(height_ratio, width_ratio+0.2)
		width_ratio = min(width_ratio,height_ratio+0.2)

                        
		if apex is not None:
			h_start, h_end, w_start, w_end = apex
		else:
			h_start, h_end, w_start, w_end = self.get_random_rectangle_inside(image.shape, height_ratio,
			               width_ratio)
		new_images = image[:, :, h_start: h_end, w_start: w_end]

            
		scaled_images = F.interpolate(
		 new_images,
		 size=[image.shape[2], image.shape[3]],
		 mode='bilinear')

		return scaled_images, (h_start, h_end, w_start, w_end)