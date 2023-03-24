from model.ds_net import SlimResNet, SlimBlock
from model.dyn_slim.dyn_slim_ops import DSLinear
import torch.nn as nn
from torch.cuda.amp import autocast
import torch
class AVnet_Slim(nn.Module):
    def __init__(self, model='resnet'):
        super(AVnet_Slim, self).__init__()
        self.model = model
        if model == 'resnet':
            self.audio = SlimResNet(SlimBlock, [3, 4, 6, 3])
            self.image = SlimResNet(SlimBlock, [3, 4, 6, 3])
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
                audio = blk_a(audio)
                image = blk_i(image)
            audio = torch.flatten(self.audio.avgpool(audio), 1)
            image = torch.flatten(self.image.avgpool(image), 1)
            print(audio.shape, image.shape)
            x = self.head(torch.cat([audio, image], dim=1))
            return x