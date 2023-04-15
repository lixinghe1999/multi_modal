import time
import torch.nn as nn
import torch
from torch.cuda.amp import autocast
from convnext_model import AdaConvNeXt
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

class HARnet(nn.Module):
    def __init__(self, pretrained=False):
        super(HARnet, self).__init__()
        embed_dim = 768
        # image, imu, radar
        self.branch = nn.ModuleList()
        self.branch.append(AdaConvNeXt(sparse_ratio=[0.8, 0.6, 0.4], pruning_loc=[3,6,9], num_classes=10, depths=[3, 3, 27, 3]))
        self.branch.append(AdaConvNeXt(sparse_ratio=[0.8, 0.6, 0.4], pruning_loc=[3,6,9], num_classes=10, depths=[3, 3, 27, 3]))
        self.branch.append(AdaConvNeXt(sparse_ratio=[0.8, 0.6, 0.4], pruning_loc=[3,6,9], num_classes=10, depths=[3, 3, 27, 3]))

        if pretrained:
            pass
        self.head = nn.Sequential(nn.Linear(embed_dim * 2, 309))
    def fusion_parameter(self):
        parameter = [{'params': self.head.parameters()},]
        return parameter

    @autocast()
    def forward(self, inputs):
        feat = []
        for x, branch in zip(inputs, self.branch):
            x = branch.preproces(x)
            x, mask, decisions = self.main_stage(x)
            x, featmap = self.final(x, mask)
            feat.append(x)
        output = self.head(torch.cat(feat))
        return output
