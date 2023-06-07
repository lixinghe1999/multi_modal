'''
backbone:
just late fusion
'''
import torch.nn as nn
import torch
from torch.cuda.amp import autocast
from .vit_model import AudioTransformer, VisionTransformer

class MBT(nn.Module):
    def __init__(self, scale='base', pretrained=True, num_class=309, modality=['flow', 'image']):
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
            self.head_verb = nn.Sequential(nn.Linear(self.embed_dim * len(modality), 1024), nn.ReLU(), nn.Linear(1024, num_class[0]))
            self.head_verb = nn.Sequential(nn.Linear(self.embed_dim * len(modality), 1024), nn.ReLU(), nn.Linear(1024, num_class[1]))
            # self.head_verb = nn.Linear(self.embed  _dim * len(modality), num_class[0])
            # self.head_noun = nn.Linear(self.embed_dim * len(modality), num_class[1])
            self.multi_head = True
        self.modality_weight = []
        self.modality = modality


    @autocast()
    def forward(self, x1, x2):
        # x = self.forward_audio_image(x1, x2)
        x = self.forward_flow_image(x1, x2)
        self.modality_weight = []
        if self.multi_head:
            verb = self.head_verb(x)
            noun = self.head_noun(x)
            for fc in [self.head_verb, self.head_noun]:
                for i in range(len(self.modality)):
                    self.modality_weight.append(nn.functional.linear(x[:, i * self.embed_dim: (i+1) * self.embed_dim], fc.weight[:, i * self.embed_dim: (i+1) * self.embed_dim], fc.bias/2))
    
            return {'verb': verb, 'noun': noun}
        else:
            self.modality_weight = [nn.functional.linear(x[:, :self.embed_dim], self.head.weight[:, :self.embed_dim],
                                                        self.head.bias/2),
                                    nn.functional.linear(x[:, self.embed_dim:], self.head.weight[:, self.embed_dim:],
                                                        self.head.bias/2)]
            x = self.head(x)
            return x
    def forward_audio_image(self, audio, image):
        audio = audio.view(-1, 1, 256, 256)
        image = image.view(-1, 3, 224, 224)
        
        B, audio = self.audio.preprocess(audio)
        B, image = self.image.preprocess(image)
        for i, (blk_a, blk_i) in enumerate(zip(self.audio.blocks, self.image.blocks)):
            audio = blk_a(audio)
            image = blk_i(image)
        audio = self.audio.norm(audio)
        image = self.image.norm(image)
        x = torch.cat([audio[:, 0], image[:, 0]], dim=1)
        return x
    def forward_flow_image(self, flow, image):
        flow = flow.view(-1, 2, 224, 224)
        image = image.view(-1, 3, 224, 224)
        
        B, flow = self.flow.preprocess(flow)
        B, image = self.image.preprocess(image)
        for i, (blk_f, blk_i) in enumerate(zip(self.flow.blocks, self.image.blocks)):
            flow = blk_f(flow)
            image = blk_i(image)
        flow = self.flow.norm(flow)
        image = self.image.norm(image)
        x = torch.cat([flow[:, 0], image[:, 0]], dim=1)
        return x
      
