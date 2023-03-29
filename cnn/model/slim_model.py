import sys
sys.path.append('./cnn')
from model.ds_net import SlimResNet
from model.dyn_slim.dyn_slim_ops import DSLinear
from model.dyn_slim.dyn_slim_blocks import AVHeadGate
import torch.nn as nn
from torch.cuda.amp import autocast
import torch
class AVnet_Slim(nn.Module):
    def __init__(self, channel_split=4):
        super(AVnet_Slim, self).__init__()
        self.channel_split = channel_split
        dims = [[int(i / channel_split * d) for i in range(1, channel_split+1)] for d in [64, 128, 256, 512]]
        self.audio = SlimResNet(pretrained=False, dims=dims)
        self.image = SlimResNet(pretrained=False, dims=dims)
        self.score_predictor = nn.ModuleList([AVHeadGate([p * 4 for p in dim],
                                                     channel_gate_num=self.channel_split) for dim in dims])
        self.head = DSLinear([1024, 2048, 3072, 4096], 309)
    def fusion_parameter(self):
        parameter = [{'params': self.head.parameters()},]
        return parameter
    def set_mode(self, mode):
        self.audio.mode = mode
        self.image.mode = mode
        self.channel_choice = (-1, -1)
    @autocast()
    def forward(self, audio, image):
        audio = self.audio.preprocess(audio)
        image = self.image.preprocess(image)
        for i, (blk_a, blk_i) in enumerate(zip(self.audio.blocks, self.image.blocks)):
            self.audio.set_layer_mode(blk_a)
            self.image.set_layer_mode(blk_i)
            self.audio.set_layer_choice(blk_a, self.channel_choice[0])
            self.image.set_layer_choice(blk_i, self.channel_choice[1])

            audio = blk_a(audio)
            image = blk_i(image)

            audio, image = self.score_predictor[i](audio, image)
            print(audio.shape, image.shape)
            self.channel_choice = self.score_predictor[i].get_gate()
            print(self.channel_choice)


        audio = torch.flatten(self.audio.avgpool(audio), 1)
        image = torch.flatten(self.image.avgpool(image), 1)
        x = self.head(torch.cat([audio, image], dim=1))
        return x