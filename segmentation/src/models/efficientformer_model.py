from src.models.efficientformer_v1 import efficientformer_l1_feat
from src.models.efficientformer_v2 import efficientformerv2_s0_feat
import torch.nn as nn
import torch
from mmseg.models.necks.fpn import FPN
from mmseg.models.decode_heads.fpn_head import FPNHead

from src.models.rgb_depth_fusion import SqueezeAndExciteFusionAdd
from src.models.context_modules import get_context_module
from src.models.model_utils import ConvBNAct, Swish, Hswish

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
class DynamicRGBD(nn.Module):
    def __init__(self,
                 num_classes=37,
                 pretrained_on_imagenet=True,
                 fuse_depth_in_rgb_encoder='SE-add',
                 activation='relu',):

        super(DynamicRGBD, self).__init__()

        self.fuse_depth_in_rgb_encoder = fuse_depth_in_rgb_encoder
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

        self.encoder_rgb = efficientformerv2_s0_feat()
        self.encoder_depth = efficientformerv2_s0_feat()
        self.neck = FPN(in_channels=[32, 48, 96, 176], out_channels=256, num_outs=4)
        self.head = FPNHead(in_channels=[256, 256, 256, 256],
                            in_index=[0, 1, 2, 3],
                            feature_strides=(4, 8, 16, 32),
                            channels=128,
                            dropout_ratio=0.1,
                            num_classes=37,
                            norm_cfg=dict(type='BN', requires_grad=True),
                            align_corners=False,)
        self.upsample1 = Upsample(mode='learned-3x3-zeropad', channels=num_classes)
        self.upsample2 = Upsample(mode='learned-3x3-zeropad', channels=num_classes)
        if pretrained_on_imagenet:
            print('load the pretrained model')
            # load imagenet pretrained or segmentation pretrained
            weight = torch.load('../assets/eformer_s0_450.pth')['model']
            # weight_backbone = {k[9:]: v for k, v in weight.items() if k.split('.')[0] == 'backbone'}
            # weight = torch.load('../assets/convnext_small_1k_224.pth')['model']
            # weight = {k: v for k, v in weight.items() if k.split('.')[0] != 'head'}
            self.encoder_rgb.load_state_dict(weight, False)
            self.encoder_depth.load_state_dict(weight, False)

    def forward(self, rgb, depth):
        rgb = self.encoder_rgb(rgb)
        depth = self.encoder_depth(depth.repeat(1, 3, 1, 1))
        out = [r + d for r, d in zip(rgb, depth)]
        out = self.neck(out)
        out = self.head(out)
        out = self.upsample1(out)
        out = self.upsample2(out)
        return out
def main():
    height = 480
    width = 640

    model = DynamicRGBD(
        pretrained_on_imagenet=False)

    # print(model)

    model.eval()
    rgb_image = torch.randn(1, 3, height, width)
    depth_image = torch.randn(1, 1, height, width)

    with torch.no_grad():
        output = model(rgb_image, depth_image)
    print(output.shape)
