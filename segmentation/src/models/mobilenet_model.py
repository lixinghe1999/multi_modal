# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mobilenet import mobilenetv3_large
from src.models.rgb_depth_fusion import SqueezeAndExciteFusionAdd
from src.models.context_modules import get_context_module
from src.models.resnet import BasicBlock, NonBottleneck1D
from src.models.model_utils import ConvBNAct, Swish, Hswish


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

        channels_decoder = [128, 128, 128]
        nr_decoder_blocks = [1, 1, 1]
        self.fuse_depth_in_rgb_encoder = fuse_depth_in_rgb_encoder

        self.activation = nn.ReLU(inplace=True)

        dims = [24, 40, 112, 160]
        self.encoder_rgb = mobilenetv3_large(feature_out=True)
        self.encoder_depth = mobilenetv3_large(feature_out=True)
        self.channels_decoder_in = dims[-1]
        if pretrained_on_imagenet:
            print('load the pretrained model')
            # load imagenet pretrained or segmentation pretrained
            # weight = torch.load('../assets/mobilenetv3-large.pth')
            weight = torch.load('../assets/mobilenetv3-large-lraspp-f128')
            self.encoder_rgb.load_state_dict(weight, True)
            self.encoder_depth.load_state_dict(weight, True)

        # context module
        if 'learned-3x3' in upsampling:
            warnings.warn('for the context module the learned upsampling is '
                          'not possible as the feature maps are not upscaled '
                          'by the factor 2. We will use nearest neighbor '
                          'instead.')
            upsampling_context_module = 'nearest'
        else:
            upsampling_context_module = upsampling

        self.se_layer = nn.ModuleList([SqueezeAndExciteFusionAdd(
                                        dims[0], activation=self.activation),
                                        SqueezeAndExciteFusionAdd(
                                        dims[1], activation=self.activation),
                                        SqueezeAndExciteFusionAdd(
                                        dims[2], activation=self.activation),
                                        SqueezeAndExciteFusionAdd(
                                        dims[3], activation=self.activation)])

        self.skip_layer = nn.ModuleList([ConvBNAct(
            dims[0], channels_decoder[2],
            kernel_size=1, activation=self.activation),
            ConvBNAct(dims[1], channels_decoder[1],
                kernel_size=1,activation=self.activation),
            ConvBNAct(dims[2], channels_decoder[0],
                    kernel_size=1, activation=self.activation)
        ])

        self.context_module, channels_after_context_module = \
            get_context_module(
                context_module,
                self.channels_decoder_in,
                channels_decoder[0],
                input_size=(height // 32, width // 32),
                activation=self.activation,
                upsampling_mode=upsampling_context_module
            )

        # decoder
        self.decoder = Decoder(
            channels_in=channels_after_context_module,
            channels_decoder=channels_decoder,
            activation=self.activation,
            nr_decoder_blocks=nr_decoder_blocks,
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling,
            num_classes=num_classes
        )

    def forward(self, rgb, depth):
        depth = depth.repeat(1, 3, 1, 1)
        # rgb = self.encoder_rgb(rgb)
        # depth = self.encoder_depth(depth)

        rgb = self.encoder_rgb.forward_layer(rgb, layers=[0, 1, 2, 3])
        depth = self.encoder_rgb.forward_layer(depth, layers=[0, 1, 2, 3])
        fuse = self.se_layer[0](rgb, depth)
        skip1 = self.skip_layer[0](fuse)

        rgb = self.encoder_rgb.forward_layer(fuse, layers=[4, 5, 6])
        depth = self.encoder_rgb.forward_layer(depth, layers=[4, 5, 6])
        fuse = self.se_layer[1](rgb, depth)
        skip2 = self.skip_layer[1](fuse)

        rgb = self.encoder_rgb.forward_layer(fuse, layers=[7, 8, 9, 10, 11, 12])
        depth = self.encoder_rgb.forward_layer(depth, layers=[7, 8, 9, 10, 11, 12])
        fuse = self.se_layer[2](rgb, depth)
        skip3 = self.skip_layer[2](fuse)

        rgb = self.encoder_rgb.forward_layer(fuse, layers=[13, 14, 15])
        depth = self.encoder_rgb.forward_layer(depth, layers=[13, 14, 15])
        fuse = self.se_layer[3](rgb, depth)
        out = self.context_module(fuse)

        out = self.decoder(enc_outs=[out, skip3, skip2, skip1])
        return out


class Decoder(nn.Module):
    def __init__(self,
                 channels_in,
                 channels_decoder,
                 activation=nn.ReLU(inplace=True),
                 nr_decoder_blocks=1,
                 encoder_decoder_fusion='add',
                 upsampling_mode='bilinear',
                 num_classes=37):
        super().__init__()

        self.decoder_module_1 = DecoderModule(
            channels_in=channels_in,
            channels_dec=channels_decoder[0],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[0],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )

        self.decoder_module_2 = DecoderModule(
            channels_in=channels_decoder[0],
            channels_dec=channels_decoder[1],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[1],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )

        self.decoder_module_3 = DecoderModule(
            channels_in=channels_decoder[1],
            channels_dec=channels_decoder[2],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[2],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )
        out_channels = channels_decoder[2]

        self.conv_out = nn.Conv2d(out_channels,
                                  num_classes, kernel_size=3, padding=1)

        # upsample twice with factor 2
        self.upsample1 = Upsample(mode=upsampling_mode,
                                  channels=num_classes)
        self.upsample2 = Upsample(mode=upsampling_mode,
                                  channels=num_classes)

    def forward(self, enc_outs):
        enc_out, enc_skip_down_16, enc_skip_down_8, enc_skip_down_4 = enc_outs

        out, out_down_32 = self.decoder_module_1(enc_out, enc_skip_down_16)
        out, out_down_16 = self.decoder_module_2(out, enc_skip_down_8)
        out, out_down_8 = self.decoder_module_3(out, enc_skip_down_4)

        out = self.conv_out(out)
        out = self.upsample1(out)
        out = self.upsample2(out)

        if self.training:
            return out, out_down_8, out_down_16, out_down_32
        return out


class DecoderModule(nn.Module):
    def __init__(self,
                 channels_in,
                 channels_dec,
                 activation=nn.ReLU(inplace=True),
                 nr_decoder_blocks=1,
                 encoder_decoder_fusion='add',
                 upsampling_mode='bilinear',
                 num_classes=37):
        super().__init__()
        self.upsampling_mode = upsampling_mode
        self.encoder_decoder_fusion = encoder_decoder_fusion

        self.conv3x3 = ConvBNAct(channels_in, channels_dec, kernel_size=3,
                                 activation=activation)

        blocks = []
        for _ in range(nr_decoder_blocks):
            blocks.append(NonBottleneck1D(channels_dec,
                                          channels_dec,
                                          activation=activation)
                          )
        self.decoder_blocks = nn.Sequential(*blocks)

        self.upsample = Upsample(mode=upsampling_mode,
                                 channels=channels_dec)

        # for pyramid supervision
        self.side_output = nn.Conv2d(channels_dec,
                                     num_classes,
                                     kernel_size=1)

    def forward(self, decoder_features, encoder_features):
        out = self.conv3x3(decoder_features)
        out = self.decoder_blocks(out)

        if self.training:
            out_side = self.side_output(out)
        else:
            out_side = None

        out = self.upsample(out)

        if self.encoder_decoder_fusion == 'add':
            out += encoder_features

        return out, out_side


class Upsample(nn.Module):
    def __init__(self, mode, channels=None):
        super(Upsample, self).__init__()
        self.interp = nn.functional.interpolate

        if mode == 'bilinear':
            self.align_corners = False
        else:
            self.align_corners = None

        if 'learned-3x3' in mode:
            # mimic a bilinear interpolation by nearest neigbor upscaling and
            # a following 3x3 conv. Only works as supposed when the
            # feature maps are upscaled by a factor 2.

            if mode == 'learned-3x3':
                self.pad = nn.ReplicationPad2d((1, 1, 1, 1))
                self.conv = nn.Conv2d(channels, channels, groups=channels,
                                      kernel_size=3, padding=0)
            elif mode == 'learned-3x3-zeropad':
                self.pad = nn.Identity()
                self.conv = nn.Conv2d(channels, channels, groups=channels,
                                      kernel_size=3, padding=1)

            # kernel that mimics bilinear interpolation
            w = torch.tensor([[[
                [0.0625, 0.1250, 0.0625],
                [0.1250, 0.2500, 0.1250],
                [0.0625, 0.1250, 0.0625]
            ]]])

            self.conv.weight = torch.nn.Parameter(torch.cat([w] * channels))

            # set bias to zero
            with torch.no_grad():
                self.conv.bias.zero_()

            self.mode = 'nearest'
        else:
            # define pad and conv just to make the forward function simpler
            self.pad = nn.Identity()
            self.conv = nn.Identity()
            self.mode = mode

    def forward(self, x):
        size = (int(x.shape[2]*2), int(x.shape[3]*2))
        x = self.interp(x, size, mode=self.mode,
                        align_corners=self.align_corners)
        x = self.pad(x)
        x = self.conv(x)
        return x


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
