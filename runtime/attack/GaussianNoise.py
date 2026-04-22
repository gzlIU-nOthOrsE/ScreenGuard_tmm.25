import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets, transforms
from torchvision.utils import save_image
import math
import torchvision.transforms.functional as TF
class GaussianNoise(nn.Module):

	def __init__(self):
		super(GaussianNoise, self).__init__()

	def gaussian_noise(self, image, var):
		noise = torch.Tensor(np.random.normal(0, var ** 0.5, image.shape)).to(image.device)
		out = image + noise
		return out

	def forward(self, image_and_cover, var):
		image = image_and_cover
		return self.gaussian_noise(image, var)






























