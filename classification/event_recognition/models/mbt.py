'''
backbone:
just late fusion
'''
import torch.nn as nn
import torch
from torch.cuda.amp import autocast
from .vit_model import AudioTransformerDiffPruning, VisionTransformerDiffPruning

class MBT(nn.Module):
    def __init__(self, scale='base', pretrained=False, num_class=309):
        super(MBT, self).__init__()
        if scale == 'base':
            config = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, pruning_loc=())
            pretrained_weight = torch.load('pretrained/deit_base_patch16_224.pth')['model']
            self.embed_dim = 768
        elif scale == 'small':
            config = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, pruning_loc=())
            pretrained_weight = torch.load('pretrained/deit_small_patch16_224.pth')['model']
            self.embed_dim = 384
        else:
            config = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True, pruning_loc=())
            pretrained_weight = torch.load('pretrained/deit_tiny_patch16_224.pth')['model']
            self.embed_dim = 192

        self.audio = AudioTransformerDiffPruning(config, pretrained=pretrained_weight)
        self.audio.head = None
        self.image = VisionTransformerDiffPruning(**config)
        if pretrained:
            self.image.load_state_dict(pretrained_weight, strict=False)
        self.image.head = None

        self.num_class = num_class
        if isinstance(num_class, int):
            self.head = nn.Linear(self.embed_dim * 2, num_class)
            self.multi_head = False
        else:
            self.head_verb = nn.Linear(self.embed_dim * 2, num_class[0])
            self.head_noun = nn.Linear(self.embed_dim * 2, num_class[1])
            self.multi_head = True
        self.modality_weight = []
    def fusion_parameter(self):
        parameter = [{'params': self.head.parameters()},
        ]
        return parameter

    @autocast()
    def forward(self, audio, image):
        if self.multi_head:
            B = audio.shape[0]
            audio = audio.view(-1, 256, 256)
            image = image.view(-1, 3, 224, 224)
        self.modality_weight = []
        _, audio = self.audio.preprocess(audio.unsqueeze(1))
        _, image = self.image.preprocess(image)

        for i, (blk_a, blk_i) in enumerate(zip(self.audio.blocks, self.image.blocks)):
            audio = blk_a(audio)
            image = blk_i(image)
        audio = self.audio.norm(audio)
        image = self.image.norm(image)

        if self.multi_head:
            audio = audio[:, 0].view(B, 3, self.embed_dim).mean(dim=1)
            image = image[:, 0].view(B, 3, self.embed_dim).mean(dim=1)
            x = torch.cat([audio, image], dim=1)
            x = torch.flatten(x, start_dim=1)
            return {'verb': self.head_verb(x), 'noun': self.head_noun(x)}
        else:
            x = torch.cat([audio[:, 0], image[:, 0]], dim=1)
            x = torch.flatten(x, start_dim=1)
            self.modality_weight = [nn.functional.linear(x[:, :self.embed_dim], self.head.weight[:, :self.embed_dim],
                                                        self.head.bias/2),
                                    nn.functional.linear(x[:, self.embed_dim:], self.head.weight[:, self.embed_dim:],
                                                        self.head.bias/2)]
            x = self.head(x)
            return x
