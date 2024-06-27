import torch
import torch.nn as nn
from einops import rearrange

class Net(nn.Module):
    def __init__(self, upscale_factor):
        super(Net, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        
        # Initial convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        
        # Final convolutional layer with output channels as upscale_factor squared times input channels
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=upscale_factor ** 2, kernel_size=3, padding=1)
        
        self.upscale_factor = upscale_factor

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle_einops(self.conv4(x))
        return x

    def pixel_shuffle_einops(self, input):
        upscale_factor = self.upscale_factor
        batch_size, channels, in_height, in_width = input.size()
        channels //= (upscale_factor * upscale_factor)
        out_height = in_height * upscale_factor
        out_width = in_width * upscale_factor

        input = rearrange(input, 'b (c r1 r2) h w -> b c (h r1) (w r2)', r1=upscale_factor, r2=upscale_factor)
        return input
