from model.ds_net import SlimResNet, Bottleneck
from model.dyn_slim.dyn_slim_ops import DSLinear
import torch.nn as nn
from torch.cuda.amp import autocast
import torch
class AVnet_Slim(nn.Module):
    def __init__(self, model='resnet'):
        super(AVnet_Slim, self).__init__()
        self.model = model
        if model == 'resnet':
            dims = [[int(0.25 * d), int(0.5 * d), int(0.75 * d), int(1 * d)] for d in [64, 128, 256, 512]]
            self.audio = SlimResNet(dims=dims)
            self.image = SlimResNet(dims=dims)
            self.head = DSLinear([1024, 2048, 3072, 4096], 309)
            # self.head = nn.Sequential(nn.Linear(embed_dim * 2, 309))
    def fusion_parameter(self):
        parameter = [{'params': self.head.parameters()},
                     #{'params': self.fusion.parameters()}
        ]
        return parameter

    @autocast()
    def forward(self, audio, image):
        if self.model == 'resnet':
            audio = self.audio.preprocess(audio)
            image = self.image.preprocess(image)
            for i, (blk_a, blk_i) in enumerate(zip(self.audio.blocks, self.image.blocks)):
                self.audio.set_layer_choice(blk_a)
                self.audio.set_layer_mode(blk_a)
                self.image.set_layer_choice(blk_i)
                self.image.set_layer_mode(blk_i)
                audio = blk_a(audio)
                audio = self.audio.score_predictor[i](audio)
                image = blk_i(image)
                image = self.image.score_predictor[i](image)

            audio = torch.flatten(self.audio.avgpool(audio), 1)
            image = torch.flatten(self.image.avgpool(image), 1)
            x = self.head(torch.cat([audio, image], dim=1))
            return x