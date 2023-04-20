# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import warnings

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from src.models.convnext import AdaConvNeXt, ConvNeXt
from src.models.rgb_depth_fusion import SqueezeAndExciteFusionAdd
from src.models.context_modules import get_context_module
from src.models.resnet import BasicBlock, NonBottleneck1D
from src.models.model_utils import ConvBNAct, Swish, Hswish
from src.models.decoder import UPerHead, FCNHead


class ConvNextRGBD(nn.Module):
    def __init__(self,
                 height=480,
                 width=640,
                 num_classes=37,
                 channels_decoder=None,  # default: [128, 128, 128]
                 pretrained_on_imagenet=True,
                 activation='relu',
                 encoder_decoder_fusion='None',
                 context_module='ppm',
                 nr_decoder_blocks=None,  # default: [1, 1, 1]
                 fuse_depth_in_rgb_encoder='SE-add',
                 upsampling='bilinear'):

        super(ConvNextRGBD, self).__init__()

        if channels_decoder is None:
            channels_decoder = [128, 128, 128]
        if nr_decoder_blocks is None:
            nr_decoder_blocks = [1, 1, 1]

        self.fuse_depth_in_rgb_encoder = fuse_depth_in_rgb_encoder

        # set activation function
        if activation.lower() == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation.lower() in ['swish', 'silu']:
            self.activation = Swish()
        elif activation.lower() == 'hswish':
            self.activation = Hswish()
        else:
            raise NotImplementedError(
                'Only relu, swish and hswish as activation function are '
                'supported so far. Got {}'.format(activation))

        dims = [96, 192, 384, 768]
        self.encoder_rgb = ConvNeXt(dims=dims, depths=[3, 3, 27, 3])
        self.encoder_depth = ConvNeXt(dims=dims, depths=[3, 3, 27, 3])
        self.UPerHead = UPerHead(
                                 # in_channels=[128, 256, 512, 1024],
                                 in_channels=[96, 192, 384, 768],
                                 in_index=[0, 1, 2, 3],
                                 pool_scales=(1, 2, 3, 6),
                                 channels=512,
                                 dropout_ratio=0.1,
                                 num_classes=37,
                                 norm_cfg=dict(type='BN', requires_grad=True),
                                 align_corners=False, )

        if pretrained_on_imagenet:
            # load imagenet pretrained or segmentation pretrained
            weight = torch.load('../assets/upernet_convnext_small_1k_512x512.pth')['state_dict']
            weight_backbone = {k[9:]: v for k, v in weight.items() if k.split('.')[0] == 'backbone'}
            # weight = torch.load('../assets/convnext_small_1k_224.pth')['model']
            # weight = {k: v for k, v in weight.items() if k.split('.')[0] != 'head'}
            self.encoder_rgb.load_state_dict(weight_backbone)
            self.encoder_depth.load_state_dict(weight_backbone)
            weight_uperhead = {k[12:]: v for k, v in weight.items() if k.split('.')[0] == 'decode_head'}
            weight_uperhead = {k: v for k, v in weight_uperhead.items() if k.split('.')[0] != 'conv_seg'}
            self.UPerHead.load_state_dict(weight_uperhead, strict=False)
        self.channels_decoder_in = dims[-1]

        if fuse_depth_in_rgb_encoder == 'SE-add':
            self.se_layer1 = SqueezeAndExciteFusionAdd(
                dims[0],
                activation=self.activation)
            self.se_layer2 = SqueezeAndExciteFusionAdd(
                dims[1],
                activation=self.activation)
            self.se_layer3 = SqueezeAndExciteFusionAdd(
                dims[2],
                activation=self.activation)
            self.se_layer4 = SqueezeAndExciteFusionAdd(
                dims[3],
                activation=self.activation)

        if encoder_decoder_fusion == 'add':
            layers_skip1 = list()
            if dims[0] != channels_decoder[2]:
                layers_skip1.append(ConvBNAct(
                    dims[0],
                    channels_decoder[2],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer1 = nn.Sequential(*layers_skip1)

            layers_skip2 = list()
            if dims[1] != channels_decoder[1]:
                layers_skip2.append(ConvBNAct(
                    dims[1],
                    channels_decoder[1],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer2 = nn.Sequential(*layers_skip2)

            layers_skip3 = list()
            if dims[2] != channels_decoder[0]:
                layers_skip3.append(ConvBNAct(
                    dims[2],
                    channels_decoder[0],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer3 = nn.Sequential(*layers_skip3)

        elif encoder_decoder_fusion == 'None':
            self.skip_layer0 = nn.Identity()
            self.skip_layer1 = nn.Identity()
            self.skip_layer2 = nn.Identity()
            self.skip_layer3 = nn.Identity()

        # context module
        # if 'learned-3x3' in upsampling:
        #     warnings.warn('for the context module the learned upsampling is '
        #                   'not possible as the feature maps are not upscaled '
        #                   'by the factor 2. We will use nearest neighbor '
        #                   'instead.')
        #     upsampling_context_module = 'nearest'
        # else:
        #     upsampling_context_module = upsampling
        # self.context_module, channels_after_context_module = \
        #     get_context_module(
        #         context_module,
        #         self.channels_decoder_in,
        #         channels_decoder[0],
        #         input_size=(height // 32, width // 32),
        #         activation=self.activation,
        #         upsampling_mode=upsampling_context_module
        #     )
        #
        # # decoder
        # self.decoder = Decoder(
        #     channels_in=channels_after_context_module,
        #     channels_decoder=channels_decoder,
        #     activation=self.activation,
        #     nr_decoder_blocks=nr_decoder_blocks,
        #     encoder_decoder_fusion=encoder_decoder_fusion,
        #     upsampling_mode=upsampling,
        #     num_classes=num_classes
        # )

    def forward(self, rgb, depth):
        # block 1
        rgb = self.encoder_rgb.forward_layer1(rgb)
        depth = self.encoder_depth.forward_layer1(depth.repeat(1, 3, 1, 1))
        if self.fuse_depth_in_rgb_encoder == 'SE-add':
            rgb, depth_t = self.se_layer1(rgb, depth)
            fuse = rgb + depth_t
        else:
            fuse = rgb + depth
        skip1 = self.skip_layer1(fuse)

        # block 2
        rgb = self.encoder_rgb.forward_layer2(fuse)
        depth = self.encoder_depth.forward_layer2(depth)
        if self.fuse_depth_in_rgb_encoder == 'SE-add':
            rgb, depth_t = self.se_layer2(rgb, depth)
            fuse = rgb + depth_t
        else:
            fuse = rgb + depth
        skip2 = self.skip_layer2(fuse)

        # block 3
        rgb = self.encoder_rgb.forward_layer3(fuse)
        depth = self.encoder_depth.forward_layer3(depth)
        if self.fuse_depth_in_rgb_encoder == 'SE-add':
            rgb, depth_t = self.se_layer3(rgb, depth)
            fuse = rgb + depth_t
        else:
            fuse = rgb + depth
        skip3 = self.skip_layer3(fuse)

        # block 4
        rgb = self.encoder_rgb.forward_layer4(fuse)
        depth = self.encoder_depth.forward_layer4(depth)
        if self.fuse_depth_in_rgb_encoder == 'SE-add':
            rgb, depth_t = self.se_layer4(rgb, depth)
            fuse = rgb + depth_t
        else:
            fuse = rgb + depth
        # out = self.context_module(fuse)
        # out = self.decoder(enc_outs=[out, skip3, skip2, skip1])
        print(skip1.shape, skip2.shape, skip3.shape, fuse.shape)
        out = self.UPerHead([skip1, skip2, skip3, fuse])
        print(out.shape)
        return out

class ConvNextOneModality(nn.Module):
    def __init__(self,
                 height=480,
                 width=640,
                 num_classes=37,
                 channels_decoder=None,  # default: [128, 128, 128]
                 pretrained_on_imagenet=True,
                 activation='relu',
                 encoder_decoder_fusion='add',
                 context_module='ppm',
                 nr_decoder_blocks=None,  # default: [1, 1, 1]
                 upsampling='bilinear'):

        super(ConvNextOneModality, self).__init__()

        if channels_decoder is None:
            channels_decoder = [128, 128, 128]
        if nr_decoder_blocks is None:
            nr_decoder_blocks = [1, 1, 1]

        # set activation function
        if activation.lower() == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation.lower() in ['swish', 'silu']:
            self.activation = Swish()
        elif activation.lower() == 'hswish':
            self.activation = Hswish()
        else:
            raise NotImplementedError(
                'Only relu, swish and hswish as activation function are '
                'supported so far. Got {}'.format(activation))

        dims = [96, 192, 384, 768]
        self.encoder = ConvNeXt(dims=dims, depths=[3, 3, 27, 3])
        if pretrained_on_imagenet:
            # load imagenet pretrained or segmentation pretrained
            # weight = torch.load('../assets/upernet_convnext_small_1k_512x512.pth')['state_dict']
            # weight = {k[9:]: v for k, v in weight.items() if k.split('.')[0] == 'backbone'}
            weight = torch.load('../assets/convnext_small_1k_224.pth')['model']
            weight = {k: v for k, v in weight.items() if k.split('.')[0] != 'head'}
            self.encoder.load_state_dict(weight)
        self.channels_decoder_in = dims[-1]

        if encoder_decoder_fusion == 'add':
            layers_skip1 = list()
            if dims[0] != channels_decoder[2]:
                layers_skip1.append(ConvBNAct(
                    dims[0],
                    channels_decoder[2],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer1 = nn.Sequential(*layers_skip1)

            layers_skip2 = list()
            if dims[1] != channels_decoder[1]:
                layers_skip2.append(ConvBNAct(
                    dims[1],
                    channels_decoder[1],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer2 = nn.Sequential(*layers_skip2)

            layers_skip3 = list()
            if dims[2] != channels_decoder[0]:
                layers_skip3.append(ConvBNAct(
                    dims[2],
                    channels_decoder[0],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer3 = nn.Sequential(*layers_skip3)

        elif encoder_decoder_fusion == 'None':
            self.skip_layer0 = nn.Identity()
            self.skip_layer1 = nn.Identity()
            self.skip_layer2 = nn.Identity()
            self.skip_layer3 = nn.Identity()

        # context module
        if 'learned-3x3' in upsampling:
            warnings.warn('for the context module the learned upsampling is '
                          'not possible as the feature maps are not upscaled '
                          'by the factor 2. We will use nearest neighbor '
                          'instead.')
            upsampling_context_module = 'nearest'
        else:
            upsampling_context_module = upsampling
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

    # @autocast()
    def forward(self, x):
        # block 1
        x = self.encoder.forward_layer1(x)
        skip1 = self.skip_layer1(x)

        # block 2
        x = self.encoder.forward_layer2(x)
        skip2 = self.skip_layer2(x)

        # block 3
        x = self.encoder.forward_layer3(x)
        skip3 = self.skip_layer3(x)

        # block 4
        x = self.encoder.forward_layer4(x)

        out = self.context_module(x)
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

    model = ConvNextRGBD(
        height=height,
        width=width,
        pretrained_on_imagenet=True)

    # print(model)

    model.eval()
    rgb_image = torch.randn(1, 3, height, width)
    depth_image = torch.randn(1, 1, height, width)

    with torch.no_grad():
        output = model(rgb_image, depth_image)
    print(output.shape)


if __name__ == '__main__':
    main()
