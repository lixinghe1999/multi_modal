from model.ds_net import SlimResNet, Bottleneck
from model.dyn_slim.dyn_slim_ops import DSLinear
import torch.nn as nn
from torch.cuda.amp import autocast
import torch
class AVnet_Slim(nn.Module):
    def __init__(self, model='resnet', channel_split=4):
        super(AVnet_Slim, self).__init__()
        self.channel_split = channel_split
        self.model = model
        if model == 'resnet':
            dims = [[int(i / channel_split * d) for i in range(1, channel_split+1)] for d in [64, 128, 256, 512]]
            self.audio = SlimResNet(pretrained=False, dims=dims)
            self.image = SlimResNet(pretrained=False, dims=dims)
            self.head = DSLinear([1024, 2048, 3072, 4096], 309)
    def fusion_parameter(self):
        parameter = [{'params': self.head.parameters()},]
        return parameter

    @autocast()
    def forward(self, audio, image):
        if self.model == 'resnet':
            comp = 0
            audio = self.audio.preprocess(audio)
            image = self.image.preprocess(image)
            for i, (blk_a, blk_i) in enumerate(zip(self.audio.blocks, self.image.blocks)):
                self.audio.set_layer_mode(blk_a)
                self.audio.set_layer_choice(blk_a)
                self.image.set_layer_mode(blk_i)
                self.image.set_layer_choice(blk_i)
                comp += (self.audio.channel_choice + self.image.channel_choice) / 8
                audio = blk_a(audio)
                # audio = self.audio.score_predictor[i](audio)
                image = blk_i(image)
                # image = self.image.score_predictor[i](image)
            comp /= 4
            audio = torch.flatten(self.audio.avgpool(audio), 1)
            image = torch.flatten(self.image.avgpool(image), 1)
            x = self.head(torch.cat([audio, image], dim=1))
            return x, comp