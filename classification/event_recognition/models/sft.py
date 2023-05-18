'''
backbone:
just late fusion
'''
import torch.nn as nn
import torch
from torch.cuda.amp import autocast
from .vit_model import AudioTransformerDiffPruning, VisionTransformerDiffPruning, Block

class SFT(nn.Module):
    def __init__(self, scale='base', pretrained=False):
        super(SFT, self).__init__()
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

        self.depth_clip = 9
        self.len_blocks = len(self.audio.blocks)
        self.image.blocks = self.image.blocks[:self.depth_clip]
        self.audio.blocks = self.audio.blocks[:self.depth_clip]
        self.sparse_blocks = nn.ModuleList([
            Block(dim=self.embed_dim, num_heads=config['num_heads'], mlp_ratio=config['mlp_ratio'], qkv_bias=config['qkv_bias'])
            for _ in range(self.len_blocks - self.depth_clip)])

        self.head = nn.Linear(self.embed_dim * 2, 309)
        self.modality_weight = []
    def fusion_parameter(self):
        parameter = [{'params': self.head.parameters()},
        ]
        return parameter

    @autocast()
    def forward(self, audio, image):
        self.modality_weight = []
        B, audio = self.audio.preprocess(audio.unsqueeze(1))
        B, image = self.image.preprocess(image)

        for i, (blk_a, blk_i) in enumerate(zip(self.audio.blocks, self.image.blocks)):
            audio = blk_a(audio)
            image = blk_i(image)

        # sparse
        x = torch.stack([audio[:, 0], image[:, 0], audio[:, 1:].mean(dim=1), image[:, 1:].mean(dim=1)], dim=2).permute(0,2,1)

        # sparse encoding
        for i, blk in enumerate(self.sparse_blocks):
            x = blk(x)
        x = self.audio.norm(x)
        x = torch.cat([x[:, 0], x[:, 1]], dim=1)
        x = torch.flatten(x, start_dim=1)
        self.modality_weight = [nn.functional.linear(x[:, :self.embed_dim], self.head.weight[:, :self.embed_dim],
                                                     self.head.bias/2),
                                nn.functional.linear(x[:, self.embed_dim:], self.head.weight[:, self.embed_dim:],
                                                     self.head.bias/2)]
        x = self.head(x)
        return x
