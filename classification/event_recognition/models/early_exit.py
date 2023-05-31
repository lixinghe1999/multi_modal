'''
Baseline:
add early-exit on each block
Two modality will always have same computation
'''
from typing import Iterator
import torch.nn as nn
import torch
from torch.cuda.amp import autocast
from torch.nn.parameter import Parameter
class Early_Exit(nn.Module):
    def __init__(self, backbone, scale='base', pretrained=False, num_class=309):
        super(Early_Exit, self).__init__()
        backbone = backbone(scale, pretrained, num_class)
        module_list = ['audio', 'image', 'embed_dim', 'head', 'multi_head', 'head_verb', 'head_noun']
        for m in module_list:
            try:
                setattr(self, m, getattr(backbone, m))
            except:
                print('Careful, do not have', m)
                pass
        self.len_blocks = len(self.audio.blocks)
        self.num_patches = self.audio.num_patches + self.image.num_patches
        self.num_class = num_class
        if isinstance(num_class, int):
            self.head = nn.ModuleList([nn.Linear(self.embed_dim * 2, num_class) for _ in range(self.len_blocks)])
            self.multi_head = False
        else:
            self.head_verb = nn.ModuleList([nn.Linear(self.embed_dim * 2, num_class[0]) for _ in range(self.len_blocks)])
            self.head_noun = nn.ModuleList([nn.Linear(self.embed_dim * 2, num_class[1]) for _ in range(self.len_blocks)])
            self.multi_head = True
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return super().parameters(recurse)
    def output(self, audio, image, i):
        audio = self.audio.norm(audio)
        image = self.image.norm(image)
        if self.multi_head:
            x = torch.cat([audio[:, 0], image[:, 0]], dim=1)
            verb = self.head_verb[i](x)
            noun = self.head_noun[i](x)
            return {'verb': verb, 'noun': noun}
        else:
            x = torch.cat([audio[:, 0], image[:, 0]], dim=1)
            x = self.head[i](x)
            return x
    @autocast()
    def forward(self, audio, image, exit=12):
        audio = audio.view(-1, 1, 256, 256)
        image = image.view(-1, 3, 224, 224)
        B, audio = self.audio.preprocess(audio)
        B, image = self.image.preprocess(image)
        output = []
        for i, (blk_a, blk_i) in enumerate(zip(self.audio.blocks, self.image.blocks)):
            audio = blk_a(audio)
            image = blk_i(image)
            o = self.output(audio, image, i)
            output.append(o)
            if i == exit:
                break
        return output

