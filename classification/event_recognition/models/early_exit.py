'''
Baseline:
add early-exit on each block
Two modality will always have same computation
'''
import torch.nn as nn
import torch
from torch.cuda.amp import autocast
class Early_Exit(nn.Module):
    def __init__(self, backbone, scale='base', pretrained=False):
        super(Early_Exit, self).__init__()
        backbone = backbone(scale, pretrained)
        module_list = ['audio', 'image', 'embed_dim']
        for m in module_list:
            setattr(self, m, getattr(backbone, m))
        self.len_blocks = len(self.audio.blocks)
        self.head = nn.ModuleList([nn.Linear(self.embed_dim * 2, 309) for _ in range(self.len_blocks)])

    def get_parameters(self):
        parameter = [{'params': self.head.parameters()}]
        return parameter

    @autocast()
    def forward(self, audio, image):
        B, audio = self.audio.preprocess(audio.unsqueeze(1))
        B, image = self.image.preprocess(image)
        output = []
        for i, (blk_a, blk_i) in enumerate(zip(self.audio.blocks, self.image.blocks)):
            audio = blk_a(audio)
            image = blk_i(image)
            audio_norm = self.audio.norm(audio)
            image_norm = self.image.norm(image)
            x = torch.cat([audio_norm[:, 0], image_norm[:, 0]], dim=1)
            x = torch.flatten(x, start_dim=1)
            output.append(self.head[i](x))
        return output

