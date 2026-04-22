import torch
import torch.nn as nn
from einops import repeat
import numpy as np


class Checkboard_Encoder(nn.Module):
    def __init__(self, input_channel=16, output_channel=3):
        super(Checkboard_Encoder, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, 3, padding=1, bias=True),
            nn.Conv2d(input_channel, input_channel * 2, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(input_channel * 2, input_channel * 4, 3, stride=2, padding=1),
            nn.Conv2d(input_channel * 4, input_channel * 4, 1, stride=1, bias=True),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(input_channel * 4, input_channel * 4, 1),
            nn.Conv2d(input_channel * 4, input_channel * 2, 3, padding=1, bias=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(input_channel * 2, input_channel * 2, 3, padding=1, bias=True),
            nn.Conv2d(input_channel * 2, input_channel, 3, padding=1, bias=True),
            nn.Conv2d(input_channel, output_channel, 1, stride=1, bias=True),
            nn.LeakyReLU()
        )

    def forward(self, x):
        
        x = self.conv_layers(x)
        return x


if __name__ == '__main__':
    batch_size = 4
    len_bits = 16
    secret_msg = np.random.randint(0, 2, size=(batch_size, len_bits)).astype(
        np.float32)  
    secret_msg = torch.from_numpy(secret_msg)
    
    secret_msg = repeat(secret_msg, 'b c -> b c h w', h=256, w=256)
    

    
    model = Checkboard_Encoder()

    
    output = model(secret_msg)

    
    print(output.shape)