'''
Baseline0:
'''
import time

import torch.nn as nn
import torch
from torch.cuda.amp import autocast
from model.vit_model import AudioTransformerDiffPruning, VisionTransformerDiffPruning
from model.resnet_model import ResNet, Bottleneck
class MMTM(nn.Module):
      def __init__(self, dim_visual, dim_skeleton, ratio):
        super(MMTM, self).__init__()
        dim = dim_visual + dim_skeleton
        dim_out = int(2*dim/ratio)
        self.fc_squeeze = nn.Linear(dim, dim_out)

        self.fc_visual = nn.Linear(dim_out, dim_visual)
        self.fc_skeleton = nn.Linear(dim_out, dim_skeleton)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

      def forward(self, visual, skeleton):
        squeeze_array = []
        for tensor in [visual, skeleton]:
          tview = tensor.view(tensor.shape[:2] + (-1,))
          squeeze_array.append(torch.mean(tview, dim=-1))
        squeeze = torch.cat(squeeze_array, 1)

        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)

        vis_out = self.fc_visual(excitation)
        sk_out = self.fc_skeleton(excitation)

        vis_out = self.sigmoid(vis_out)
        sk_out = self.sigmoid(sk_out)

        dim_diff = len(visual.shape) - len(vis_out.shape)
        vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)

        dim_diff = len(skeleton.shape) - len(sk_out.shape)
        sk_out = sk_out.view(sk_out.shape + (1,) * dim_diff)

        return visual * vis_out, skeleton * sk_out

class AVnet_Early(nn.Module):
    def __init__(self, model='resnet', pretrained=False):
        super(AVnet_Early, self).__init__()
        self.model = model
        if model == 'resnet':
            self.net = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=6)
            embed_dim = 512 * 4
            self.head = nn.Sequential(nn.Linear(embed_dim, 309))
        else:
            config = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                          pruning_loc=())
            embed_dim = 768
            self.audio = AudioTransformerDiffPruning(config, imagenet_pretrain=pretrained)
            self.image = VisionTransformerDiffPruning(**config)
            if pretrained:
                self.image.load_state_dict(torch.load('assets/deit_base_patch16_224.pth')['model'], strict=False)
            self.head = nn.Sequential(nn.Linear(embed_dim * 2, 309))
    def fusion_parameter(self):
        parameter = [{'params': self.head.parameters()},
                     #{'params': self.fusion.parameters()}
        ]
        return parameter

    @autocast()
    def forward(self, audio, image):
        if self.model == 'resnet':
            print(audio.shape, image.shape)
            x = torch.cat([audio, image])
            x = self.net.preprocess(x)
            for i, blk in enumerate(self.net.blocks):
                x = blk(x)
                # audio, image = self.fusion[i](audio, image)
            x = torch.flatten(self.net.avgpool(x), 1)
            x = self.head(x)
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
