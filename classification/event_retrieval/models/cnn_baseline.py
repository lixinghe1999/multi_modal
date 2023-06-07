'''
backbone:
just late fusion
'''
import torch.nn as nn
import torch
from torch.cuda.amp import autocast
import torchvision

class CNN(nn.Module):
    def __init__(self, scale='resnet', pretrained=True, num_class=309, modality=['audio', 'image']):
        super(CNN, self).__init__()
        import copy
        if scale == 'mobilenet':
            backbone = torchvision.models.mobilenet_v3_large(pretrained=pretrained)
        else:
            backbone = torchvision.models.resnet152(pretrained=pretrained)
        if 'audio' in modality:
            self.audio = copy.deepcopy(backbone)
            self.audio.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if 'image' in modality:
            self.image = backbone
        self.embed_dim = 1000

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
        audio = self.audio(audio)
        image = self.image(image)
        self.modality_weight = []
        if self.multi_head:
            x = torch.cat([audio, image], dim=1)
            verb = self.head_verb(x)
            noun = self.head_noun(x)
            for fc in [self.head_verb, self.head_noun]:
                for i in range(len(self.modality)):
                    self.modality_weight.append(nn.functional.linear(x[:, i * self.embed_dim: (i+1) * self.embed_dim], fc.weight[:, i * self.embed_dim: (i+1) * self.embed_dim], fc.bias/2))
    
            return {'verb': verb, 'noun': noun}
        else:
            x = torch.cat([audio, image], dim=1)
            self.modality_weight = [nn.functional.linear(x[:, :self.embed_dim], self.head.weight[:, :self.embed_dim],
                                                        self.head.bias/2),
                                    nn.functional.linear(x[:, self.embed_dim:], self.head.weight[:, self.embed_dim:],
                                                        self.head.bias/2)]
            x = self.head(x)
            return x
