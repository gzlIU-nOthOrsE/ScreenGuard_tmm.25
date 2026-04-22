import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image


class Cropout(nn.Module):
	"""
	Remove a random region of the original image, the region is a rectangle
	"""
	def __init__(self):
		super(Cropout, self).__init__()

	def get_random_rectangle_inside(self, image_shape, area_ratio):
		'''返回随机的长方形区域(返回顶点坐标)'''
		
		image_height = image_shape[2]
		image_width = image_shape[3]

		height_ratio = np.random.uniform(area_ratio, 1)
		width_ratio = area_ratio / height_ratio

		cropout_height = int(height_ratio * image_height)
		cropout_width = int(width_ratio * image_width)

		
		if cropout_height == image_height:
			height_start = 0
		else:
			height_start = np.random.randint(0, image_height - cropout_height)

		if cropout_width == image_width:
			width_start = 0
		else:
			width_start = np.random.randint(0, image_width - cropout_width)

		return height_start, height_start + cropout_height, width_start, width_start + cropout_width

	def forward(self, image, cur_rate):
		if isinstance(cur_rate, list):
			assert len(cur_rate) == 1
			cur_rate = cur_rate[0]
		h_start, h_end, w_start, w_end = self.get_random_rectangle_inside(image.shape, cur_rate)
		mask = np.ones(image.shape)
		mask[:, :, h_start: h_end, w_start: w_end] = 0
		mask_tensor = torch.tensor(mask, device=image.device, dtype=torch.float)

		cropped_image = mask_tensor * image

		
		return cropped_image












