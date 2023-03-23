'''
Baseline0:
'''
import time

import torch.nn as nn
import torch
from torch.cuda.amp import autocast
from model.vit_model import AudioTransformerDiffPruning, VisionTransformerDiffPruning
from model.resnet_model import resnet50

class AVnet(nn.Module):
    def __init__(self, model='resnet', pretrained=False):
        super(AVnet, self).__init__()
        self.model = model
        if model == 'resnet':
            self.audio = resnet50(pretrained=pretrained)
            self.image = resnet50(pretrained=pretrained)
            embed_dim = 512 * 4
            self.head = nn.Sequential(nn.Linear(embed_dim * 2, 309))
        else:
            config = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                          pruning_loc=())
            embed_dim = 768
            self.audio = AudioTransformerDiffPruning(config, imagenet_pretrain=pretrained)
            self.image = VisionTransformerDiffPruning(**config)
            if pretrained:
                self.image.load_state_dict(torch.load('assets/deit_base_patch16_224.pth')['model'], strict=False)
            self.head = nn.Sequential(nn.LayerNorm(embed_dim * 2),
                                          nn.Linear(embed_dim * 2, 309))
    def fusion_parameter(self):
        parameter = [{'params': self.head.parameters()}]
        return parameter

    @autocast()
    def forward(self, audio, image):
        if self.model == 'resnet':
            audio = self.audio._forward_impl(audio)
            image = self.image._forward_impl(image)
            x = self.head(torch.cat([audio, image], dim=1))
            return x
        else:
            B, audio = self.audio.preprocess(audio.unsqueeze(1))
            B, image = self.image.preprocess(image)

            for i, (blk_a, blk_i) in enumerate(zip(self.audio.blocks, self.image.blocks)):
                audio = blk_a(audio)
                image = blk_i(image)
            audio = self.audio.norm(audio)
            image = self.image.norm(image)
            x = torch.cat([audio[:, 0], image[:, 0]], dim=1)
            x = torch.flatten(x, start_dim=1)
            x = self.head(x)
            return x
if __name__ == "__main__":
    device = 'cuda'
    base_rate = 0.5
    pruning_loc = [3, 6, 9]
    token_ratio = [base_rate, base_rate ** 2, base_rate ** 3]

    model = AVnet().to(device)

    model.eval()

    audio = torch.zeros(1, 384, 128).to(device)
    image = torch.zeros(1, 3, 224, 224).to(device)
    num_iterations = 100
    t_start = time.time()
    for _ in range(num_iterations):
        model(audio, image)
    print((time.time() - t_start)/num_iterations)
