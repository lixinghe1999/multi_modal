'''
backbone:
just late fusion
'''
import torch.nn as nn
import torch
from torch.cuda.amp import autocast
from .vit_model import AudioTransformerDiffPruning, VisionTransformerDiffPruning
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
from collections import OrderedDict

class MBT(nn.Module):
    def __init__(self, scale='base', pretrained=False, num_class=309):
        super(MBT, self).__init__()
        self.image = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.audio = vit_b_16(image_size=256)
        self.audio.conv_proj = nn.Conv2d(in_channels=1, out_channels=768, kernel_size=16, stride=16)
        self.image.heads = None
        self.audio.heads = None
        self.num_class = num_class
        if isinstance(num_class, int):
            self.head = nn.Linear(self.audio.hidden_dim + self.image.hidden_dim, num_class)
            self.multi_head = False
        else:
            self.head_verb = nn.Linear(self.audio.hidden_dim + self.image.hidden_dim, num_class[0])
            self.head_noun = nn.Linear(self.audio.hidden_dim + self.image.hidden_dim, num_class[1])
            self.multi_head = True
        self.modality_weight = []
        
    def fusion_parameter(self):
        parameter = [{'params': self.head.parameters()},
        ]
        return parameter

    # @autocast()
    def forward(self, audio, image):
        B = audio.shape[0]
        if self.multi_head:
            audio = audio.view(-1, 256, 256)
            image = image.view(-1, 3, 224, 224)
        self.modality_weight = []

        image = self.image._process_input(image)
        n = image.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.image.class_token.expand(n, -1, -1)
        image = torch.cat([batch_class_token, image], dim=1)
        
        audio = self.audio._process_input(audio.unsqueeze(1))
        n = audio.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.audio.class_token.expand(n, -1, -1)
        audio = torch.cat([batch_class_token, audio], dim=1)

        audio = audio + self.audio.encoder.pos_embedding
        audio = self.audio.encoder.dropout(audio)
        image = image + self.image.encoder.pos_embedding
        image = self.image.encoder.dropout(image)
        
        for layer_audio, layer_image in zip(self.audio.encoder.layers, self.image.encoder.layers):
            audio = layer_audio(audio)
            image = layer_image(image) 
        # audio = self.audio.encoder.layers(audio)
        # image = self.image.encoder.layers(image)

        audio = self.audio.encoder.ln(audio)
        image = self.image.encoder.ln(image)

        if self.multi_head:
            audio = audio[:, 0].view(B, 3, self.audio.hidden_dim).mean(dim=1)
            image = image[:, 0].view(B, 3, self.image.hidden_dim).mean(dim=1)
            x = torch.cat([audio, image], dim=1)
            verb = self.head_verb(x)
            noun = self.head_noun(x)
            return {'verb': verb, 'noun': noun}
        else:
            # print(self.audio.heads, self.image.heads)
            # audio = self.audio.heads(audio[:, 0])
            # image = self.image.heads(image[:, 0])
            # print(audio.shape, image.shape)
            # x = (audio + image)/2      
            # print(self.heads)      
            x = torch.cat([audio[:, 0], image[:, 0]], dim=1)
            x = self.heads(x)
            self.modality_weight = [nn.functional.linear(x[:, :self.audio.hidden_dim], self.head.weight[:, :self.audio.hidden_dim], self.head.bias/2),
                                    nn.functional.linear(x[:, self.image.hidden_dim:], self.head.weight[:, self.image.hidden_dim:], self.head.bias/2)]
            
            return x
       