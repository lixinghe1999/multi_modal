# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import warnings

import torch
import torch.nn as nn
from src.models.fastseg import MobileV3Large

class MobileRGBD(nn.Module):
    def __init__(self,
                 height=480,
                 width=640,
                 num_classes=37,
                 pretrained_on_imagenet=True,
                 encoder_decoder_fusion='add',
                 context_module='ppm',
                 fuse_depth_in_rgb_encoder='SE-add',
                 upsampling='bilinear'):

        super(MobileRGBD, self).__init__()

        self.encoder_rgb = MobileV3Large.from_pretrained()
        self.encoder_depth = MobileV3Large.from_pretrained()

    def forward(self, rgb, depth):
        depth = depth.repeat(1, 3, 1, 1)
        rgb = self.encoder_rgb(rgb)
        depth = self.encoder_depth(depth)
        print(rgb.shape, depth.shape)


def main():
    height = 480
    width = 640

    model = MobileRGBD(
        height=height,
        width=width)
    model.eval()
    rgb_image = torch.randn(1, 3, height, width)
    depth_image = torch.randn(1, 1, height, width)

    with torch.no_grad():
        output = model(rgb_image, depth_image)


if __name__ == '__main__':
    main()
