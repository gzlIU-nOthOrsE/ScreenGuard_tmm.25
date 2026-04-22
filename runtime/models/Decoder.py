import math
import random
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from einops import repeat
import segmentation_models_pytorch as smp


class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation='relu', kernel_initializer='he_normal'):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.kernel_initializer = kernel_initializer

        self.linear = nn.Linear(in_features, out_features)
        
        if kernel_initializer == 'he_normal':
            nn.init.kaiming_normal_(self.linear.weight)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        outputs = self.linear(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
        return outputs


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', strides=1, padding=True):
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation
        self.strides = strides
        padding = int((kernel_size - 1) / 2) if padding is True else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding=padding)
        
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.activation is not None:
            if self.activation == 'relu':
                outputs = nn.ReLU(inplace=True)(outputs)
            else:
                raise NotImplementedError
        return outputs


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.contiguous().view(input.size(0), -1)


class SpatialTransformerNetwork(nn.Module):
    def __init__(self, image_size=400):
        super(SpatialTransformerNetwork, self).__init__()
        self.localization = nn.Sequential(
            Conv2D(3, 32, 3, strides=2, activation='relu'),
            Conv2D(32, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 128, 3, strides=2, activation='relu'),
            Flatten(),
            Dense((image_size // 8) ** 2 * 128, 128, activation='relu'),
            nn.Linear(128, 6)
        )
        self.localization[-1].weight.data.fill_(0)
        self.localization[-1].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    def forward(self, image):
        theta = self.localization(image)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, image.size(), align_corners=False)
        transformed_image = F.grid_sample(image, grid, align_corners=False)
        return transformed_image



class StegaStampDecoder(nn.Module):
    def __init__(self, stn=True, secret_size=16, image_size=256):
        super(StegaStampDecoder, self).__init__()
        self.image_size = image_size
        self.secret_size = secret_size
        if stn is True:
            self.stn = SpatialTransformerNetwork(image_size=image_size)
        else:
            self.stn = None
        assert image_size == 256  
        self.decoder = nn.Sequential(
            Conv2D(3, 32, 3, strides=2, activation='relu'),
            Conv2D(32, 32, 3, activation='relu'),
            Conv2D(32, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 64, 3, activation='relu'),
            Conv2D(64, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 128, 3, strides=2, activation='relu'),
            Conv2D(128, 128, 3, strides=2, activation='relu'),
            Flatten(),
            Dense(8192, 512, activation='relu'),
            Dense(512, secret_size, activation=None))

    def forward(self, image):
        
        transformed_image = self.stn(image) if self.stn is not None else image
        extracted = self.decoder(transformed_image)
        return extracted



class BlockDecoder(nn.Module):
    def __init__(self, secret_size=1, image_size=256):
        super(BlockDecoder, self).__init__()
        self.image_size = image_size
        self.secret_size = secret_size
        assert image_size == 256  
        self.decoder = nn.Sequential(
            Conv2D(3, 32, 3, activation='relu'),
            Conv2D(32, 16, 3, activation='relu'),
            Flatten(),
            Dense(4096, 512, activation='relu'),
            Dense(512, secret_size, activation=None))

    def forward(self, image):
        
        transformed_image = image.reshape(-1, 3, 16, 16)
        extracted = self.decoder(transformed_image)
        return extracted



class ImageDecoder(nn.Module):
    def __init__(self, image_size=192, tile_size=16, do_parser=False):
        super(ImageDecoder, self).__init__()
        self.image_size = image_size
        self.tile_size = tile_size
        self.do_parser = do_parser
        assert image_size % tile_size == 0

        if self.do_parser:
            ENCODER = 'se_resnext50_32x4d'
            ENCODER_WEIGHTS = 'imagenet'
            CLASSES = ['R', 'G', 'B']
            ACTIVATION = None  
            
            self.parser = smp.UnetPlusPlus(
                encoder_name=ENCODER,
                encoder_weights=ENCODER_WEIGHTS,
                classes=len(CLASSES),
                activation=ACTIVATION,
            )

        self.decoder = nn.Sequential(
            Conv2D(3, 32, 3, activation='relu'),
            Conv2D(32, 64, 3, activation='relu'),
            Conv2D(64, 128, 3, activation='relu'),
            Conv2D(128, 64, tile_size, strides=tile_size, activation='relu'),
            Conv2D(64, 16, 1, activation='relu'),
            Conv2D(16, 1, 1, activation=None)
        )

    def forward(self, image):
        if self.do_parser:
            image = self.parser(image)
        out = self.decoder(image)
        return out



class FilterLayer(nn.Module):
    def __init__(self):
        super(FilterLayer, self).__init__()
        self.miner = nn.Sequential(
            Conv2D(3, 32, 3, activation='relu'),
            Conv2D(32, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 64, 3, activation='relu'),
            Conv2D(64, 64, 3, activation='relu'),
            Conv2D(64, 128, 3, activation='relu'),
            Conv2D(128, 128, 3, activation='relu'),
            Conv2D(128, 3, 1, activation=None)
        )

    def forward(self, image):
        return self.miner(image)



class HeatMapDecoder(nn.Module):
    def __init__(self):
        super(HeatMapDecoder, self).__init__()
        
        self.miner = nn.Sequential(
            Conv2D(3, 16, 3, activation='relu'),
            Conv2D(16, 32, 3, activation='relu'),
            Conv2D(32, 32, 3, activation='relu'),
            Conv2D(32, 64, 3, activation='relu'),
        )
        self.lt_locator = nn.Sequential(
            Conv2D(64, 16, kernel_size=16, strides=1, activation='relu', padding=False),
            Conv2D(16, 1, 1, activation=None)
        )
        self.rt_locator = nn.Sequential(
            Conv2D(64, 16, kernel_size=16, strides=1, activation='relu', padding=False),
            Conv2D(16, 1, 1, activation=None)
        )
        self.ld_locator = nn.Sequential(
            Conv2D(64, 16, kernel_size=16, strides=1, activation='relu', padding=False),
            Conv2D(16, 1, 1, activation=None)
        )
        self.finer = nn.Sequential(
            Conv2D(3, 16, 17, activation='relu', strides=1, padding=True),
            Conv2D(16, 3, 1, activation=None)
        )

    def forward(self, image):
        features = self.miner(image)
        lt_heatmap = self.lt_locator(features)
        rt_heatmap = self.rt_locator(features)
        ld_heatmap = self.ld_locator(features)
        out = torch.cat([lt_heatmap, rt_heatmap, ld_heatmap], dim=1)
        out = self.finer(out)
        return out



class BCDecoder(nn.Module):
    def __init__(self):
        super(BCDecoder, self).__init__()
        self.decoder = nn.Sequential(
            Conv2D(3, 32, 3, activation='relu'),
            Conv2D(32, 32, 5, activation='relu'),
            Conv2D(32, 64, 5, activation='relu'),
            Conv2D(64, 64, 3, activation='relu'),
            Conv2D(64, 64, 3, activation='relu'),
            Conv2D(64, 128, 3, activation='relu'),
            Conv2D(128, 64, 1, activation='relu'),
            Conv2D(64, 16, 1, activation='relu'),
            Conv2D(16, 1, 1, activation=None),
        )

    def forward(self, image):
        extracted = self.decoder(image)
        return extracted


if __name__ == '__main__':
    batch_size = 4
    len_bits = 3
    secret_msg = np.random.randint(0, 2, size=(batch_size, len_bits)).astype(
        np.float32)  
    secret_msg = torch.from_numpy(secret_msg)
    
    secret_msg = repeat(secret_msg, 'b c -> b c h w', h=256, w=256)

    t = HeatMapDecoder()
    to = t(secret_msg)
    print(to.shape)