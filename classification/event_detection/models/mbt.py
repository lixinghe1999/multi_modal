'''
backbone:
just late fusion
'''
import torch.nn as nn
import torch
from torch.cuda.amp import autocast
from .vit_model import AudioTransformer, VisionTransformer

class MBT(nn.Module):
    def __init__(self, scale='base', pretrained=True, num_class=309, modality=['audio', 'image']):
        super(MBT, self).__init__()
        if scale == 'base':
            pretrained_weight = 'pretrained/deit_base_patch16_224.pth'
            config = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True)
            self.embed_dim = 768
        elif scale == 'small':
            pretrained_weight = 'pretrained/deit_small_patch16_224.pth'
            config = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True)
            self.embed_dim = 384
        else:
            pretrained_weight = 'pretrained/deit_tiny_patch16_224.pth'
            config = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True)
            self.embed_dim = 192
        if pretrained:
            pretrained_weight = torch.load(pretrained_weight)['model']
            config['pretrained'] = pretrained_weight
        if 'audio' in modality:
            self.audio = AudioTransformer(config)
            self.audio.head = None
        if 'image' in modality:
            self.image = VisionTransformer(**config)
            self.image.head = None
        if 'flow' in modality:
            self.flow = VisionTransformer(**config, in_chans=2)
            self.flow.head = None

        self.num_class = num_class
        if isinstance(num_class, int):
            self.head = nn.Linear(self.embed_dim * len(modality), num_class)
            self.multi_head = False
        else:
            self.head_verb = nn.Linear(self.embed_dim * len(modality), num_class[0])
            self.head_noun = nn.Linear(self.embed_dim * len(modality), num_class[1])
            self.multi_head = True
        self.modality_weight = []
        self.modality = modality


    @autocast()
    def forward(self, audio, image):
        B = audio.shape[0]
        audio = audio.view(-1, 1, 256, 256)
        image = image.view(-1, 3, 224, 224)
        
        _, audio = self.audio.preprocess(audio)
        _, image = self.image.preprocess(image)
        for i, (blk_a, blk_i) in enumerate(zip(self.audio.blocks, self.image.blocks)):
            audio = blk_a(audio)
            image = blk_i(image)
        audio = self.audio.norm(audio)
        image = self.image.norm(image)
        self.modality_weight = []
        if self.multi_head:
            x = torch.cat([audio[:, 0], image[:, 0]], dim=1)
            verb = self.head_verb(x)
            noun = self.head_noun(x)
            for fc in [self.head_verb, self.head_noun]:
                for i in range(len(self.modality)):
                    self.modality_weight.append(nn.functional.linear(x[:, i * self.embed_dim: (i+1) * self.embed_dim], fc.weight[:, i * self.embed_dim: (i+1) * self.embed_dim], fc.bias/2))
    
            return {'verb': verb, 'noun': noun}
        else:
            x = torch.cat([audio[:, 0], image[:, 0]], dim=1)
            self.modality_weight = [nn.functional.linear(x[:, :self.embed_dim], self.head.weight[:, :self.embed_dim],
                                                        self.head.bias/2),
                                    nn.functional.linear(x[:, self.embed_dim:], self.head.weight[:, self.embed_dim:],
                                                        self.head.bias/2)]
            x = self.head(x)
            return x
